# expert-agents/github_issue_fetcher_agent.py
import re
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, Any

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext # For after_tool_callback
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
    "Additional context"
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

# THIS CALLBACK IS REINSTATED AND CORRECTED
def github_issue_fetcher_after_tool_callback(
    tool: BaseTool, # The tool that was just run by github_issue_fetcher_agent's LLM (i.e., AgentTool(browser_agent))
    args: dict,     # Args passed to that tool
    tool_context: ToolContext, # Context of github_issue_fetcher_agent
    tool_response: Any # Output from AgentTool(agent=browser_agent)
) -> genai_types.Content | None:
    # This callback runs after github_issue_fetcher_agent's LLM calls AgentTool(browser_agent).
    # The 'tool' here is AgentTool(browser_agent), its name is browser_agent.name.
    if tool.name == browser_agent.name:
        logger.info(f"FetcherAgent (after_tool_callback for {browser_agent.name}): Processing browser's output. Raw response type: {type(tool_response)}, value: {str(tool_response)[:500]}...")
        
        # AgentTool(agent=browser_agent) should return the direct text output of browser_agent's
        # own after_tool_callback, which is '{"extracted_details": "RAW_CONTENT_FROM_BROWSER_TOOL"}'.
        json_string_from_browser_agent_output = None

        if isinstance(tool_response, str):
            json_string_from_browser_agent_output = tool_response
        elif isinstance(tool_response, dict) and "text" in tool_response: # Common AgentTool wrapping
            json_string_from_browser_agent_output = tool_response["text"]
        elif isinstance(tool_response, dict) and "result" in tool_response: # Alternative AgentTool wrapping
            result_val = tool_response["result"]
            if isinstance(result_val, genai_types.Content) and result_val.parts and result_val.parts[0].text:
                json_string_from_browser_agent_output = result_val.parts[0].text
            elif isinstance(result_val, str):
                json_string_from_browser_agent_output = result_val
        
        if not json_string_from_browser_agent_output:
            logger.warning(f"FetcherAgent (after_tool_callback): Could not extract JSON string from {browser_agent.name}'s output: {tool_response}")
            final_payload = {"error": f"Browser utility ({browser_agent.name}) returned an unreadable response."}
            return genai_types.Content(parts=[genai_types.Part(text=json.dumps(final_payload))])

        raw_extracted_details = ""
        try:
            # Expect '{"extracted_details": "..."}'
            parsed_browser_output = json.loads(json_string_from_browser_agent_output)
            if isinstance(parsed_browser_output, dict) and "extracted_details" in parsed_browser_output:
                raw_extracted_details = parsed_browser_output["extracted_details"]
                logger.info("FetcherAgent (after_tool_callback): Successfully parsed 'extracted_details' from browser output.")
            elif isinstance(parsed_browser_output, dict) and "error" in parsed_browser_output: # Error from browser_agent_after_tool_callback
                 logger.error(f"FetcherAgent (after_tool_callback): Browser agent itself reported an error: {parsed_browser_output['error']}")
                 final_payload = {"error": parsed_browser_output['error']}
                 return genai_types.Content(parts=[genai_types.Part(text=json.dumps(final_payload))])
            else: 
                logger.warning(f"FetcherAgent (after_tool_callback): Parsed JSON from browser was not expected dict. Parsed: {str(parsed_browser_output)[:200]}")
                raw_extracted_details = json_string_from_browser_agent_output # Fallback
        except json.JSONDecodeError:
            logger.warning(f"FetcherAgent (after_tool_callback): Output from browser was not JSON. Using as raw. Output: {json_string_from_browser_agent_output[:200]}")
            raw_extracted_details = json_string_from_browser_agent_output
        
        final_payload_dict = {}
        if raw_extracted_details and "Error:" not in raw_extracted_details: # Basic check for error string
            cleaned_details = clean_github_issue_text(raw_extracted_details)
            if not cleaned_details:
                final_payload_dict = {"message": "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."}
            else:
                # THIS IS THE DESIRED OUTPUT STRUCTURE FOR ROOT_AGENT
                final_payload_dict = {"cleaned_issue_details": cleaned_details}
        else: 
            error_message = "The browser tool could not fetch or extract content from the GitHub issue."
            if raw_extracted_details and "Error:" in raw_extracted_details:
                 error_message = raw_extracted_details 
            final_payload_dict = {"error": error_message}

        final_json_output_for_root_agent = json.dumps(final_payload_dict)
        logger.info(f"FetcherAgent (after_tool_callback): Returning direct JSON content to root_agent: {final_json_output_for_root_agent[:200]}...")
        # By returning Content here, github_issue_fetcher_agent's LLM is bypassed.
        # This Content becomes the final output of github_issue_fetcher_agent for this step.
        tool_context.actions.skip_summarization = True # Reinforce finality
        return genai_types.Content(parts=[genai_types.Part(text=final_json_output_for_root_agent)])
    
    logger.warning(f"FetcherAgent (after_tool_callback): Callback triggered for unexpected tool: {tool.name}")
    return None

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    issue_number_str = None
    invocation_ctx = context._invocation_context if hasattr(context, '_invocation_context') else None

    if invocation_ctx and invocation_ctx.user_content and invocation_ctx.user_content.parts:
        first_part_text = invocation_ctx.user_content.parts[0].text
        if first_part_text:
            try:
                input_data = GitHubIssueFetcherToolInput.model_validate_json(first_part_text)
                issue_number_str = input_data.issue_number
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider): Error parsing input '{first_part_text}': {e}")
                # This agent's LLM will output this JSON string.
                return "Your final response must be the JSON: {\"error\": \"Fetcher agent received invalid input. Expected an issue number.\"}"
    
    if issue_number_str:
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Instructing LLM to call browser_agent for URL: {github_url}")
        # This instruction tells the FetcherAgent's LLM to make a tool call.
        # The after_tool_callback will then intercept the browser_agent's output.
        return f"Your ONLY task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. Do not add any other text or commentary. The output of that tool will be processed by a callback."
    else:
        logger.error("FetcherAgent (instruction_provider): No issue_number. Should have been caught by root_agent.")
        return "Your final response must be the JSON: {\"error\": \"Fetcher agent was called without an issue number by the orchestrator.\"}"

github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content, cleans it, and returns a JSON object with 'cleaned_issue_details', 'message', or 'error'.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) 
    ],
    input_schema=GitHubIssueFetcherToolInput,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=github_issue_fetcher_after_tool_callback, # REINSTATED
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=512, # LLM only makes a tool call
        top_p=0.6
    )
)