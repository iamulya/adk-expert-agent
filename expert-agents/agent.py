# expert-agents/agent.py
import json
import os
import logging
import re # Make sure re is imported
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool

from google.genai import types as genai_types

from .context_loader import get_escaped_adk_context_for_llm
from .github_issue_fetcher_agent import github_issue_fetcher_agent, GitHubIssueFetcherInput
from .adk_guidance_agent import adk_guidance_agent, AdkGuidanceInput # AdkGuidanceInput only needs document_text
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    cleaned_issue_details_from_fetcher = None
    fetcher_asked_for_issue_number = False
    fetcher_reported_empty = False # New flag

    if hasattr(context, '_invocation_context') and \
       hasattr(context._invocation_context, 'session') and \
       context._invocation_context.session and \
       context._invocation_context.session.events and \
       len(context._invocation_context.session.events) > 1:
        
        potential_tool_response_event = context._invocation_context.session.events[-1]
        potential_tool_call_event = context._invocation_context.session.events[-2]

        if potential_tool_call_event.author == root_agent.name and \
           potential_tool_call_event.get_function_calls() and \
           potential_tool_call_event.get_function_calls()[0].name == github_issue_fetcher_agent.name:
            
            if potential_tool_response_event.get_function_responses():
                tool_response_part = potential_tool_response_event.get_function_responses()[0]
                if tool_response_part.name == github_issue_fetcher_agent.name:
                    tool_output_content_text = ""
                    if isinstance(tool_response_part.response, dict) and "text" in tool_response_part.response:
                        tool_output_content_text = tool_response_part.response["text"]
                    elif isinstance(tool_response_part.response, str):
                        tool_output_content_text = tool_response_part.response

                    if "Please provide the GitHub issue number" in tool_output_content_text:
                        fetcher_asked_for_issue_number = True
                        logger.info("RootAgent: Fetcher agent asked for issue number. Relaying to user.")
                    elif "The fetched GitHub issue content appears to be empty" in tool_output_content_text:
                        fetcher_reported_empty = True # Set flag
                        # This is a final message to relay, no need to call guidance agent
                        cleaned_issue_details_from_fetcher = tool_output_content_text 
                        logger.info("RootAgent: Fetcher agent reported empty/boilerplate content. Relaying to user.")
                    elif "---BEGIN CLEANED ISSUE TEXT---" in tool_output_content_text:
                        match = re.search(r"---BEGIN CLEANED ISSUE TEXT---\n(.*?)\n---END CLEANED ISSUE TEXT---", tool_output_content_text, re.DOTALL)
                        if match:
                            cleaned_issue_details_from_fetcher = match.group(1).strip()
                            logger.info(f"RootAgent: Received cleaned issue details: {cleaned_issue_details_from_fetcher[:100]}...")
                        else:
                            logger.warning("RootAgent: Could not parse cleaned issue details from fetcher agent response.")
                            cleaned_issue_details_from_fetcher = "Error: Could not parse details from fetcher. Will not proceed to guidance."
                            fetcher_reported_empty = True # Treat parsing failure as an end state for this flow
                    # If it's just some other text, it might be an error or unexpected output from fetcher
                    elif tool_output_content_text: 
                        logger.warning(f"RootAgent: Fetcher returned unexpected text: {tool_output_content_text[:200]}. Treating as end of GitHub flow.")
                        cleaned_issue_details_from_fetcher = tool_output_content_text # Relay this unexpected message
                        fetcher_reported_empty = True # Treat as an end state

    user_query_text = get_text_from_content(context.user_content) if context.user_content else ""

    # Condition to call adk_guidance_agent
    if cleaned_issue_details_from_fetcher and not fetcher_asked_for_issue_number and not fetcher_reported_empty:
        logger.info("RootAgent: Preparing to call adk_guidance_agent.")
        # Construct the exact JSON string for the 'document_text' argument
        # The LLM needs to create a JSON string like: {"document_text": "..."}
        # We escape the cleaned_issue_details_from_fetcher to be safely embedded in a JSON string within the prompt.
        escaped_cleaned_details = json.dumps(cleaned_issue_details_from_fetcher)

        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
You have received cleaned content which needs ADK-specific guidance.
Your task is to call the '{adk_guidance_agent.name}' tool.
You MUST pass the following exact cleaned content as the 'document_text' argument to the tool.
The argument should be a JSON object like this: {{"document_text": "THE_CLEANED_CONTENT_HERE"}}.
The cleaned content is:
{cleaned_issue_details_from_fetcher}

Construct the 'document_text' argument for the tool using this cleaned content.
This is your only action for this turn. After calling the tool, its response will be the final answer to the user.
Do not add any conversational fluff before calling the tool.
"""
    # Condition to call github_issue_fetcher_agent (initial query or user provided issue number)
    # Or if fetcher asked for issue number, root agent's job is to relay that.
    elif fetcher_asked_for_issue_number:
        logger.info("RootAgent: Fetcher asked for issue number. Relaying to user.")
        system_instruction = "The previous agent asked for a GitHub issue number. Please relay this request: 'Please provide the GitHub issue number for google/adk-python that you would like me to look into.' This is your final response for this turn."
    elif fetcher_reported_empty: # If fetcher agent already determined content is empty/boilerplate
        logger.info("RootAgent: Fetcher reported empty/boilerplate. Relaying its message.")
        system_instruction = f"Please relay the following message to the user: '{cleaned_issue_details_from_fetcher}' This is your final response for this turn."
    elif "github" in user_query_text.lower() or \
         any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature request"]):
        logger.info("RootAgent: Detected GitHub-related query. Calling github_issue_fetcher_agent.")
        # The LLM needs to construct a JSON string like: {"user_query": "THE_USER_QUERY_HERE"}
        escaped_user_query = json.dumps(user_query_text)
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user's query seems to be about a GitHub issue.
Your primary action is to call the '{github_issue_fetcher_agent.name}' tool.
You MUST pass the user's query as the 'user_query' argument to the tool.
The argument should be a JSON object like this: {{"user_query": "THE_USER_QUERY_HERE"}}.
The user's query is: "{user_query_text}"

Construct the 'user_query' argument for the tool using the user's query.
The tool will respond. You will then take that response and decide the next step in a subsequent turn.
This is your only action for this turn. Do not add any conversational fluff before calling the tool.
"""
    # General ADK question
    else:
        logger.info("RootAgent: Detected general ADK query.")
        system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 0.5.0.
Your primary role is to answer general questions about ADK.
When a user starts a conversation, greet them by introducing yourself as an ADK 0.5.0 expert.
Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general ADK questions):
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
"""
    return system_instruction

API_KEY = get_gemini_api_key_from_secret_manager()

root_agent = ADKAgent(
    name="adk_expert_orchestrator",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=root_agent_instruction_provider,
    tools=[
        AgentTool(agent=github_issue_fetcher_agent),
        AgentTool(agent=adk_guidance_agent)
    ],
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=60000,
        top_p=0.6,
    )
)