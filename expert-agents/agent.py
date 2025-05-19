# expert-agents/agent.py
import os
import logging
import re
import json # Ensure json is imported
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool

from google.genai import types as genai_types

from .context_loader import get_escaped_adk_context_for_llm
from .github_issue_fetcher_agent import github_issue_fetcher_agent, GitHubIssueFetcherInput
from .adk_guidance_agent import adk_guidance_agent, AdkGuidanceInput
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
    fetcher_reported_empty = False
    guidance_agent_was_called = False # New flag
    guidance_agent_response_text = None # To store guidance agent's response

    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context

    if invocation_ctx and hasattr(invocation_ctx, 'session') and \
       invocation_ctx.session and invocation_ctx.session.events and \
       len(invocation_ctx.session.events) > 1:
        
        potential_tool_response_event = invocation_ctx.session.events[-1]
        potential_tool_call_event = invocation_ctx.session.events[-2]

        if potential_tool_call_event.author == root_agent.name and \
           potential_tool_call_event.get_function_calls():
            
            called_tool_name = potential_tool_call_event.get_function_calls()[0].name

            if called_tool_name == github_issue_fetcher_agent.name:
                if potential_tool_response_event.get_function_responses():
                    tool_response_part = potential_tool_response_event.get_function_responses()[0]
                    if tool_response_part.name == github_issue_fetcher_agent.name:
                        tool_output_content_text = ""
                        # AgentTool usually returns a dict with "text" for simple text outputs
                        if isinstance(tool_response_part.response, dict) and "text" in tool_response_part.response:
                            tool_output_content_text = tool_response_part.response["text"]
                        elif isinstance(tool_response_part.response, str): # Direct string output
                            tool_output_content_text = tool_response_part.response

                        if "Please provide the GitHub issue number" in tool_output_content_text:
                            fetcher_asked_for_issue_number = True
                            logger.info("RootAgent: Fetcher agent asked for issue number.")
                        elif "The fetched GitHub issue content appears to be empty" in tool_output_content_text:
                            fetcher_reported_empty = True
                            cleaned_issue_details_from_fetcher = tool_output_content_text 
                            logger.info("RootAgent: Fetcher agent reported empty/boilerplate content.")
                        elif "---BEGIN CLEANED ISSUE TEXT---" in tool_output_content_text:
                            match = re.search(r"---BEGIN CLEANED ISSUE TEXT---\n(.*?)\n---END CLEANED ISSUE TEXT---", tool_output_content_text, re.DOTALL)
                            if match:
                                cleaned_issue_details_from_fetcher = match.group(1).strip()
                                logger.info(f"RootAgent: Received cleaned issue details: {cleaned_issue_details_from_fetcher[:100]}...")
                            else:
                                logger.warning("RootAgent: Could not parse cleaned issue details from fetcher response.")
                                cleaned_issue_details_from_fetcher = "Error: Could not parse details from fetcher."
                                fetcher_reported_empty = True
                        elif tool_output_content_text: 
                            logger.warning(f"RootAgent: Fetcher returned unexpected text: {tool_output_content_text[:200]}.")
                            cleaned_issue_details_from_fetcher = tool_output_content_text 
                            fetcher_reported_empty = True
            
            elif called_tool_name == adk_guidance_agent.name:
                guidance_agent_was_called = True
                if potential_tool_response_event.get_function_responses():
                    tool_response_part = potential_tool_response_event.get_function_responses()[0]
                    if tool_response_part.name == adk_guidance_agent.name:
                        # Guidance agent output (via AgentTool) should be in response['text']
                        if isinstance(tool_response_part.response, dict) and "text" in tool_response_part.response:
                            guidance_agent_response_text = tool_response_part.response["text"]
                            logger.info(f"RootAgent: Received response from adk_guidance_agent: {guidance_agent_response_text[:100]}...")
                        elif isinstance(tool_response_part.response, str): # If it somehow returned a direct string
                             guidance_agent_response_text = tool_response_part.response
                             logger.info(f"RootAgent: Received direct string response from adk_guidance_agent: {guidance_agent_response_text[:100]}...")
                        else:
                            logger.error(f"RootAgent: Unexpected response format from adk_guidance_agent: {tool_response_part.response}")
                            guidance_agent_response_text = "Error: Could not understand the guidance provided."


    user_query_text = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        user_query_text = get_text_from_content(invocation_ctx.user_content)

    # --- Orchestration Logic ---
    if guidance_agent_was_called:
        logger.info("RootAgent: Guidance agent was called. Presenting its response as final answer.")
        # **FIX 1**: Directly use the guidance agent's response.
        system_instruction = f"""
You have received a response from the ADK Guidance Agent.
This is the final answer for the user.
Present the following text to the user, exactly as it is, without any modifications, summarization, or additional conversational fluff:
--- ADK GUIDANCE RESPONSE ---
{guidance_agent_response_text or "No guidance was available."}
--- END ADK GUIDANCE RESPONSE ---
"""
    elif cleaned_issue_details_from_fetcher and not fetcher_asked_for_issue_number and not fetcher_reported_empty:
        logger.info("RootAgent: Preparing to call adk_guidance_agent.")
        # **FIX 2**: Ensure the LLM only passes the cleaned details for document_text
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
You have received cleaned content which needs ADK-specific guidance.
Your task is to call the '{adk_guidance_agent.name}' tool.
The tool expects a JSON argument with a single key 'document_text'.
The value for 'document_text' MUST BE EXACTLY the following cleaned content, and nothing else:
--- CLEANED CONTENT TO PASS ---
{cleaned_issue_details_from_fetcher}
--- END CLEANED CONTENT TO PASS ---

Construct the JSON argument for the tool like this: {{"document_text": "THE_CLEANED_CONTENT_FROM_ABOVE_HERE"}}.
This is your only action for this turn. The response from this tool will be the final answer.
Do not add any conversational fluff before calling the tool.
"""
    elif fetcher_asked_for_issue_number:
        logger.info("RootAgent: Fetcher asked for issue number. Relaying to user.")
        system_instruction = "The previous agent asked for a GitHub issue number. Your final response for this turn MUST be exactly: 'Please provide the GitHub issue number for google/adk-python that you would like me to look into.'"
    elif fetcher_reported_empty:
        logger.info("RootAgent: Fetcher reported empty/boilerplate. Relaying its message.")
        system_instruction = f"Your final response for this turn MUST be exactly: '{cleaned_issue_details_from_fetcher}'"
    elif "github" in user_query_text.lower() or \
         any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature request"]):
        logger.info("RootAgent: Detected GitHub-related query. Calling github_issue_fetcher_agent.")
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user's query is: "{user_query_text}"
Your primary action is to call the '{github_issue_fetcher_agent.name}' tool.
The tool expects a JSON argument with a single key 'user_query'.
The value for 'user_query' MUST BE EXACTLY the user's query provided above.
Construct the JSON argument like this: {{"user_query": "THE_USER_QUERY_FROM_ABOVE_HERE"}}.
This is your only action for this turn. The tool will respond, and you will decide the next step based on that response in a subsequent turn.
Do not add any conversational fluff before calling the tool.
"""
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