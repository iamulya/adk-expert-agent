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
    tool_context: ToolContext, 
    tool_response: Any
) -> genai_types.Content | None:
    if tool.name == browser_agent.name:
        logger.info(f"FetcherAgent (after_tool_callback): Processing response from tool '{tool.name}'. Raw response type: {type(tool_response)}, value: {str(tool_response)[:500]}...")
        
        json_string_from_browser_agent_tool_output = None
        if isinstance(tool_response, dict):
            if "text" in tool_response and isinstance(tool_response["text"], str):
                json_string_from_browser_agent_tool_output = tool_response["text"]
            elif "result" in tool_response:
                result_val = tool_response["result"]
                if isinstance(result_val, genai_types.Content) and result_val.parts and \
                   isinstance(result_val.parts[0], genai_types.Part) and result_val.parts[0].text:
                    json_string_from_browser_agent_tool_output = result_val.parts[0].text
                elif isinstance(result_val, str):
                    json_string_from_browser_agent_tool_output = result_val
        elif isinstance(tool_response, str):
            json_string_from_browser_agent_tool_output = tool_response
        
        if not json_string_from_browser_agent_tool_output:
            logger.warning(f"FetcherAgent (after_tool_callback): Could not extract usable JSON string from tool_response: {tool_response}")
            # Return a JSON string indicating error for root_agent to parse
            error_payload = json.dumps({"error": "Browser utility returned an unreadable response."})
            return genai_types.Content(parts=[genai_types.Part(text=error_payload)])

        raw_extracted_details = ""
        try:
            # Expect '{"extracted_details": "..."}' from browser_agent_after_tool_callback
            parsed_output = json.loads(json_string_from_browser_agent_tool_output)
            if isinstance(parsed_output, dict) and "extracted_details" in parsed_output:
                raw_extracted_details = parsed_output["extracted_details"]
                logger.info("FetcherAgent (after_tool_callback): Successfully parsed 'extracted_details'.")
            else: 
                logger.warning(f"FetcherAgent (after_tool_callback): Parsed JSON was not the expected dict with 'extracted_details'. Parsed: {str(parsed_output)[:200]}")
                raw_extracted_details = json_string_from_browser_agent_tool_output # Fallback
        except json.JSONDecodeError:
            logger.warning("FetcherAgent (after_tool_callback): Output from browser agent wrapper was not JSON. Using as raw details (might be an error msg).")
            raw_extracted_details = json_string_from_browser_agent_tool_output
        
        final_payload_dict = {}
        if raw_extracted_details and "Error:" not in raw_extracted_details : 
            cleaned_details = clean_github_issue_text(raw_extracted_details)
            if not cleaned_details:
                final_payload_dict = {"message": "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."}
            else:
                final_payload_dict = {"cleaned_issue_details": cleaned_details}
        else: 
            error_message = "The browser tool could not fetch or extract content from the GitHub issue."
            if raw_extracted_details and "Error:" in raw_extracted_details:
                 error_message = raw_extracted_details 
            final_payload_dict = {"error": error_message}

        final_json_output_for_root_agent = json.dumps(final_payload_dict)
        logger.info(f"FetcherAgent (after_tool_callback): Returning direct JSON content: {final_json_output_for_root_agent[:200]}...")
        tool_context.actions.skip_summarization = True 
        return genai_types.Content(parts=[genai_types.Part(text=final_json_output_for_root_agent)])
    
    logger.warning(f"FetcherAgent (after_tool_callback): Callback triggered for unexpected tool: {tool.name}")
    return None

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    # This instruction provider is now only for the LLM to make the browser_agent tool call
    # if it receives an issue_number. The actual processing of the browser's output
    # is handled by the after_tool_callback.

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
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider): Could not parse issue_number from input '{first_part_text}': {e}")
                return "INTERNAL ERROR: This agent received invalid input. Expected an issue number."
    
    if issue_number_str:
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Instructing LLM to call browser_agent for URL: {github_url}")
        return f"Your ONLY task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. Do not add any other text or commentary."
    else:
        logger.error("FetcherAgent (instruction_provider): No issue_number was provided to fetch. This indicates an error in the calling agent's logic.")
        # This agent should always be called with an issue_number by the root_agent now.
        # Returning an error message that root_agent can relay if this path is hit.
        return json.dumps({"error": "Fetcher agent was called without an issue number."})


github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content using a browser, and returns a JSON object containing either 'cleaned_issue_details', a 'message', or an 'error'.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent)
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