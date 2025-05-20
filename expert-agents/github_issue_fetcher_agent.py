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
    "Describe alternatives you'veconsidered",
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
    if not text: return ""
    cleaned_text = text
    for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(boilerplate, "")
    return re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()

class GitHubIssueFetcherToolInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext, 
    tool_response: Any  # Expected: '{"status": "BROWSER_DATA_READY"}' or '{"error": "..."}' or potentially empty string
) -> genai_types.Content | None:
    if tool.name == browser_agent.name:
        logger.info(f"FetcherAgent (after_tool_callback for {tool.name}): Received tool_response: '{str(tool_response)[:150]}'.")

        browser_agent_llm_output_parsed = None
        browser_agent_signaled_ready = False
        browser_agent_reported_error = None

        # Try to parse tool_response (sentinel or error from browser_agent's LLM)
        if isinstance(tool_response, str) and tool_response.strip():
            try:
                browser_agent_llm_output_parsed = json.loads(tool_response)
                if isinstance(browser_agent_llm_output_parsed, dict):
                    if browser_agent_llm_output_parsed.get("status") == "BROWSER_DATA_READY":
                        browser_agent_signaled_ready = True
                        logger.info("FetcherAgent: Received BROWSER_DATA_READY sentinel.")
                    elif "error" in browser_agent_llm_output_parsed:
                        browser_agent_reported_error = browser_agent_llm_output_parsed["error"]
                        logger.warning(f"FetcherAgent: Browser agent's LLM reported an error: {browser_agent_reported_error}")
                    else:
                        logger.warning(f"FetcherAgent: Received unexpected JSON from browser_agent's LLM: {str(browser_agent_llm_output_parsed)[:100]}")
                else:
                    logger.warning(f"FetcherAgent: Parsed browser_agent's LLM response is not a dict: {str(browser_agent_llm_output_parsed)[:100]}")
            except json.JSONDecodeError:
                logger.warning(f"FetcherAgent: tool_response from browser_agent's LLM was not valid JSON: {str(tool_response)[:100]}")
        elif isinstance(tool_response, dict) and "text" in tool_response and isinstance(tool_response["text"], str) and tool_response["text"].strip(): # ADK AgentTool might wrap it
            try:
                browser_agent_llm_output_parsed = json.loads(tool_response["text"])
                if isinstance(browser_agent_llm_output_parsed, dict):
                    if browser_agent_llm_output_parsed.get("status") == "BROWSER_DATA_READY":
                        browser_agent_signaled_ready = True
                    elif "error" in browser_agent_llm_output_parsed:
                        browser_agent_reported_error = browser_agent_llm_output_parsed["error"]
            except json.JSONDecodeError:
                 logger.warning(f"FetcherAgent: tool_response text from browser_agent's LLM (wrapped) was not valid JSON: {str(tool_response['text'])[:100]}")
        else:
             logger.warning(f"FetcherAgent: Received empty or unhandled tool_response from browser_agent's LLM: type {type(tool_response)}, value '{str(tool_response)[:100]}'")


        # Always try to get data from session state
        actual_data_dict = tool_context._invocation_context.session.state.get("temp:browser_tool_output_data")
        
        if actual_data_dict is not None:
            tool_context._invocation_context.session.state.pop("temp:browser_tool_output_data", None)
            logger.info("FetcherAgent (after_tool_callback): Consumed 'temp:browser_tool_output_data' from session state.")
        else:
            logger.warning("FetcherAgent (after_tool_callback): 'temp:browser_tool_output_data' not found in session state.")

        final_payload_dict = {}
        if isinstance(actual_data_dict, dict):
            logger.info(f"FetcherAgent (after_tool_callback): Processing data from session state: {str(actual_data_dict)[:200]}...")
            if "extracted_details" in actual_data_dict:
                raw_extracted_details = actual_data_dict["extracted_details"]
                logger.info("FetcherAgent: Successfully processed 'extracted_details' from session state.")
                cleaned_details = clean_github_issue_text(raw_extracted_details)
                if not cleaned_details:
                    final_payload_dict = {"message": "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."}
                else:
                    final_payload_dict = {"cleaned_issue_details": cleaned_details}
            elif "error" in actual_data_dict: 
                final_payload_dict = {"error": f"Browser tool (via session state) reported: {actual_data_dict['error']}"}
            else: # Data in state, but wrong structure
                final_payload_dict = {"error": "Browser tool returned unexpected data structure via session state."}
        else: # No data in state, or invalid data type
            if browser_agent_reported_error:
                 final_payload_dict = {"error": f"Browser agent init error: {browser_agent_reported_error}. No data from browser tool."}
            elif not browser_agent_signaled_ready:
                 final_payload_dict = {"error": "Browser utility failed to signal readiness and no data was found in session."}
            else: # Signaled ready, but no data. Should be rare.
                 final_payload_dict = {"error": "Browser utility signaled ready, but no data was found in session."}
            logger.error(f"FetcherAgent: Final error payload: {final_payload_dict}")
            
        final_json_output_for_root_agent = json.dumps(final_payload_dict)
        logger.info(f"FetcherAgent (after_tool_callback): Returning direct JSON content to root_agent: {final_json_output_for_root_agent[:200]}...")
        tool_context.actions.skip_summarization = True 
        return genai_types.Content(parts=[genai_types.Part(text=final_json_output_for_root_agent)])
    
    logger.warning(f"FetcherAgent (after_tool_callback): Callback for unexpected tool: {tool.name}")
    return None

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    issue_number_str = None
    invocation_ctx = context._invocation_context if hasattr(context, '_invocation_context') else None

    if invocation_ctx and invocation_ctx.user_content and invocation_ctx.user_content.parts:
        first_part_text = invocation_ctx.user_content.parts[0].text
        if first_part_text:
            try:
                # Input from root_agent is '{"issue_number":"..."}'
                input_data = GitHubIssueFetcherToolInput.model_validate_json(first_part_text)
                issue_number_str = input_data.issue_number
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider): Error parsing input '{first_part_text}': {e}")
                return "Your final response must be the JSON: {\"error\": \"Fetcher agent received invalid input. Expected an issue number from orchestrator.\"}"
    
    if issue_number_str:
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Instructing LLM to call browser_agent for URL: {github_url}")
        
        # browser_agent now has an input_schema: BrowserAgentInput(url: str)
        # So, AgentTool will expect arguments for browser_agent to be a JSON string of that schema.
        browser_agent_args = {"url": github_url}
        browser_agent_args_json_str = json.dumps(browser_agent_args)
        
        return f"Your ONLY task is to call the '{browser_agent.name}' tool with the following JSON arguments: '{browser_agent_args_json_str}'. Do not add any other text or commentary. A callback will process the tool's output."
    else:
        logger.error("FetcherAgent (instruction_provider): No issue_number. Orchestrator should provide it.")
        return "Your final response must be the JSON: {\"error\": \"Fetcher agent was called by orchestrator without an issue number.\"}"


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
    after_tool_callback=github_issue_fetcher_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000, 
        top_p=0.6
    )
)