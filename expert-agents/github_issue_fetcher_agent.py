# expert-agents/github_issue_fetcher_agent.py
import re
import logging
import json
from typing import Any # Ensure json is imported
from pydantic import BaseModel, Field

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext 
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool 
from google.adk.tools.tool_context import ToolContext 
from google.genai import types as genai_types

from .browser_agent import browser_agent
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

# ... BOILERPLATE_STRINGS_TO_REMOVE and clean_github_issue_text ...
BOILERPLATE_STRINGS_TO_REMOVE = [
    "Is your feature request related to a problem? Please describe.",
    "Describe the solution you'd like",
    "Describe alternatives you've considered",
    "Describe the bug",
    "Minimal Reproduction",
    "Minimal steps to reproduce",
    "Desktop (please complete the following information):",
    "Please make sure you read the contribution guide and file the issues in the rigth place.",
    "To Reproduce",
    "Expected behavior",
    "Screenshots",
    "Additional context",
    "Here is the content of the GitHub issue:",
    "The content of the GitHub issue is:"
]

def clean_github_issue_text(text: str) -> str:
    if not text:
        return ""
    cleaned_text = text
    for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(boilerplate, "")
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    return cleaned_text

class GitHubIssueFetcherToolInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext, # Context of github_issue_fetcher_agent
    tool_response: Any # Output from AgentTool(agent=browser_agent)
) -> genai_types.Content | None:
    # This callback is for tools called by github_issue_fetcher_agent.
    # The only tool it calls is AgentTool(agent=browser_agent).
    if tool.name == browser_agent.name: # browser_agent.name is "browser_utility_agent"
        logger.info(f"FetcherAgent (after_tool_callback): Processing response from tool '{tool.name}'. Raw response type: {type(tool_response)}, value: {str(tool_response)[:500]}...")
        
        # AgentTool typically returns a dictionary. If its sub-agent (browser_agent)
        # returned Content(parts=[Part(text="SOME_STRING")]), AgentTool might put this
        # string under a "text" key, or sometimes under a "result" key which could
        # contain the original Content object or its textual representation.
        
        json_string_from_browser_agent = None

        if isinstance(tool_response, dict):
            if "text" in tool_response and isinstance(tool_response["text"], str):
                # This is the most common case if browser_agent's callback returned Content(text=json_string)
                # and AgentTool wrapped it.
                json_string_from_browser_agent = tool_response["text"]
            elif "result" in tool_response:
                # Sometimes AgentTool might put the sub-agent's Content object here.
                result_val = tool_response["result"]
                if isinstance(result_val, genai_types.Content) and result_val.parts and \
                   isinstance(result_val.parts[0], genai_types.Part) and result_val.parts[0].text:
                    json_string_from_browser_agent = result_val.parts[0].text
                elif isinstance(result_val, str): # Or it might have already extracted the text
                    json_string_from_browser_agent = result_val
                else:
                    logger.warning(f"FetcherAgent (after_tool_callback): 'result' key found in tool_response but not in expected format. Value: {result_val}")
            else:
                logger.warning(f"FetcherAgent (after_tool_callback): tool_response from {tool.name} is a dict but lacks 'text' or 'result' key. Dict: {tool_response}")
        elif isinstance(tool_response, str):
            # Less common for AgentTool, but handle if it directly passed the string
            json_string_from_browser_agent = tool_response
        else:
            logger.error(f"FetcherAgent (after_tool_callback): Unexpected tool_response format from {tool.name}. Expected str or dict. Got: {type(tool_response)}")
            final_text = "Error: Internal error processing browser response."
            return genai_types.Content(parts=[genai_types.Part(text=final_text)])

        if not json_string_from_browser_agent:
            logger.warning(f"FetcherAgent (after_tool_callback): Could not extract a usable string from tool_response: {tool_response}")
            final_text = "Error: Browser utility returned an unreadable response."
            return genai_types.Content(parts=[genai_types.Part(text=final_text)])

        logger.info(f"FetcherAgent (after_tool_callback): Extracted JSON string from browser agent wrapper: '{json_string_from_browser_agent[:200]}...'")
        
        raw_extracted_details = ""
        try:
            # browser_agent_after_tool_callback is designed to return a JSON string: '{"extracted_details": "..."}'
            parsed_output = json.loads(json_string_from_browser_agent)
            if isinstance(parsed_output, dict) and "extracted_details" in parsed_output:
                raw_extracted_details = parsed_output["extracted_details"]
                logger.info("FetcherAgent (after_tool_callback): Successfully parsed 'extracted_details'.")
            else: # Should not happen if browser_agent_after_tool_callback works
                logger.warning(f"FetcherAgent (after_tool_callback): Parsed JSON was not the expected dict with 'extracted_details'. Parsed: {str(parsed_output)[:200]}")
                raw_extracted_details = json_string_from_browser_agent # Fallback
        except json.JSONDecodeError:
            # This means browser_agent_after_tool_callback didn't return valid JSON.
            # This could be an error message from browser_agent itself.
            logger.warning("FetcherAgent (after_tool_callback): Output from browser agent wrapper was not JSON. Using as raw details (might be an error msg).")
            raw_extracted_details = json_string_from_browser_agent
        
        if raw_extracted_details and "Error:" not in raw_extracted_details : # Basic check for error string
            cleaned_details = clean_github_issue_text(raw_extracted_details)
            if not cleaned_details:
                final_text = "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."
            else:
                final_text = f"---BEGIN CLEANED ISSUE TEXT---\n{cleaned_details}\n---END CLEANED ISSUE TEXT---"
        else: 
            final_text = "The browser tool could not fetch or extract content from the GitHub issue."
            if raw_extracted_details and "Error:" in raw_extracted_details:
                 final_text = raw_extracted_details # Relay the specific error message from browser/fetcher

        logger.info(f"FetcherAgent (after_tool_callback): Returning direct content: {final_text[:100]}...")
        # THIS RETURN VALUE BYPASSES github_issue_fetcher_agent's LLM
        return genai_types.Content(parts=[genai_types.Part(text=final_text)])
    
    logger.warning(f"FetcherAgent (after_tool_callback): Callback triggered for unexpected tool: {tool.name}")
    return None # No action for other tools

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    issue_number_str = None
    
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context

    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content and \
       invocation_ctx.user_content.parts:
        first_part_text = invocation_ctx.user_content.parts[0].text
        if first_part_text:
            try:
                input_data = GitHubIssueFetcherToolInput.model_validate_json(first_part_text)
                issue_number_str = input_data.issue_number
                logger.info(f"FetcherAgent (instruction_provider): Received issue_number: '{issue_number_str}'.")
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider): Could not parse issue_number from input '{first_part_text}': {e}")
                return "Error: This agent expects an issue number to be provided by the calling agent."
    
    if issue_number_str:
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Instructing LLM to call browser_agent for URL: {github_url}")
        # This instruction tells the FetcherAgent's LLM to make a tool call.
        # The after_tool_callback will then intercept the browser_agent's output.
        return f"Your task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. This is your only action for this turn. Do not add any conversational text."
    else:
        logger.error("FetcherAgent (instruction_provider): No issue_number was provided or parsed. This indicates an issue with how this agent was called.")
        return "Error: No issue number was provided to fetch."

github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content using a browser, and returns the cleaned content.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) # This is browser_utility_agent
    ],
    input_schema=GitHubIssueFetcherToolInput,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=github_issue_fetcher_after_tool_callback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000, 
        top_p=0.6
    )
)