# expert-agents/github_issue_fetcher_agent.py
import re
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, Any

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext 
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from .browser_agent import browser_agent, BrowserAgentInput, BrowserAgentOutput # Import schemas
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
    if not text: return ""
    cleaned_text = text
    for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(boilerplate, "")
    return re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()

class GitHubIssueFetcherToolInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

# Helper to get text from user_content in instruction_provider
def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = context._invocation_context
    user_input_json_str = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""

    # Check if data is ready in state to be outputted to root_agent
    # This state is set by this agent's own after_tool_callback
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.state.get("temp:fetcher_json_ready_for_root"):
        json_to_output = invocation_ctx.session.state.get("temp:fetcher_json_ready_for_root")
        logger.info(f"FetcherAgent (instruction_provider): Found final JSON in state. Instructing LLM to output: {json_to_output[:100]}...")
        invocation_ctx.session.state.pop("temp:fetcher_json_ready_for_root", None) # Consume state
        return f"Your final response for this turn MUST be exactly: {json_to_output}"
    else:
        # Initial call from root_agent: instruct LLM to call browser_agent
        issue_number_str = None
        if user_input_json_str:
            try:
                input_data = GitHubIssueFetcherToolInput.model_validate_json(user_input_json_str)
                issue_number_str = input_data.issue_number
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider - initial): Error parsing input '{user_input_json_str}': {e}")
                # This agent's LLM will output this error JSON
                error_payload = {"error": "Fetcher agent received invalid input for issue_number."}
                return f"Your final response for this turn MUST be exactly: '{json.dumps(error_payload)}'"

        if not issue_number_str:
            logger.error("FetcherAgent (instruction_provider - initial): No issue_number provided.")
            error_payload = {"error": "Fetcher agent not provided with issue_number."}
            return f"Your final response for this turn MUST be exactly: '{json.dumps(error_payload)}'"

        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        # browser_agent expects BrowserAgentInput schema
        browser_agent_input_for_tool = BrowserAgentInput(url=github_url)
        browser_agent_args_json_str = browser_agent_input_for_tool.model_dump_json()
        
        logger.info(f"FetcherAgent (instruction_provider - initial): Instructing LLM to call '{browser_agent.name}' with args: {browser_agent_args_json_str}")
        return f"Your ONLY task is to call the '{browser_agent.name}' tool with the following JSON arguments: '{browser_agent_args_json_str}'. Do not add any other text or commentary. A callback will process the tool's output."


def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext, 
    tool_response: Any  # Expected: dict (parsed BrowserAgentOutput) or str (raw LLM output if parsing failed)
) -> dict | None: # Must return dict to trigger re-prompt of its own LLM
    if tool.name == browser_agent.name:
        logger.info(f"FetcherAgent (after_tool_callback for {tool.name}): Received tool_response. Type: {type(tool_response)}, Value: '{str(tool_response)[:150]}'.")

        final_payload_dict_for_root = {}

        # AgentTool with an output_schema on the sub-agent should provide a dict.
        if isinstance(tool_response, dict):
            # Attempt to validate against BrowserAgentOutput, though AgentTool might have already done this.
            # If validation fails, it means browser_agent's LLM didn't conform to its output_schema.
            try:
                # We don't strictly need to re-validate if AgentTool does it, but it's safer.
                # If AgentTool already passed a validated model's dict, this will just work.
                # If AgentTool passed a dict that doesn't match, this will raise error.
                browser_data = BrowserAgentOutput.model_validate(tool_response) 
                
                if browser_data.extracted_details:
                    cleaned_details = clean_github_issue_text(browser_data.extracted_details)
                    if not cleaned_details:
                        final_payload_dict_for_root = {"message": "GitHub issue content empty/template after cleaning."}
                    else:
                        final_payload_dict_for_root = {"cleaned_issue_details": cleaned_details}
                elif browser_data.error:
                    final_payload_dict_for_root = {"error": f"Browser agent reported: {browser_data.error}"}
                elif browser_data.message: # Handle message from browser-use if no details/error
                    final_payload_dict_for_root = {"message": f"Browser agent message: {browser_data.message}"}
                else: # Valid BrowserAgentOutput but all fields are None
                    final_payload_dict_for_root = {"message": "Browser agent returned no details, error, or message."}
            except Exception as e: # Covers Pydantic ValidationError or other issues
                logger.warning(f"FetcherAgent: tool_response dict from browser_agent couldn't be validated or processed: {str(tool_response)[:150]}. Error: {e}")
                final_payload_dict_for_root = {"error": "Browser agent returned data in an unexpected dictionary format."}
        
        # Fallback: If AgentTool passed a string (e.g., LLM failed to produce JSON for output_schema)
        elif isinstance(tool_response, str):
            logger.warning(f"FetcherAgent: tool_response from browser_agent was a string (expected dict): {str(tool_response)[:150]}. This might indicate browser_agent's LLM failed to produce valid JSON for its output_schema.")
            final_payload_dict_for_root = {"error": f"Browser agent returned unparsable text: {tool_response[:100]}..."}
        
        else: # Unexpected type for tool_response
            logger.error(f"FetcherAgent: tool_response from browser_agent was neither dict nor string. Type: {type(tool_response)}, Value: {str(tool_response)[:150]}")
            final_payload_dict_for_root = {"error": "Received unexpected data type from browser sub-process."}
        
        # Store the JSON string that this fetcher_agent's LLM needs to output to the root_agent
        json_to_output_for_root = json.dumps(final_payload_dict_for_root)
        tool_context.state["temp:fetcher_json_ready_for_root"] = json_to_output_for_root
        logger.info(f"FetcherAgent (after_tool_callback): Stored final JSON for root in state 'temp:fetcher_json_ready_for_root': {json_to_output_for_root[:100]}...")
        
        tool_context.actions.skip_summarization = True # Critical: re-prompt fetcher_agent's instruction_provider
        return {"status": "fetcher_processed_browser_data_and_result_in_state"} # Satisfy dict return type
    
    logger.warning(f"FetcherAgent (after_tool_callback): Callback for unexpected tool: {tool.name}")
    return None


github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content using a browser utility, cleans it, and returns a JSON object with 'cleaned_issue_details', 'message', or 'error'.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=GitHubIssueFetcherToolInput,
    # NO output_schema here, its LLM directly produces the final JSON string for root_agent.
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) 
    ],
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=github_issue_fetcher_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=60000, 
        top_p=0.6
    )
)