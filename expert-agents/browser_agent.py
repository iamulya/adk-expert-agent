# expert-agents/browser_agent.py
import logging
import json
from pydantic import BaseModel
from typing import Optional

from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types

from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

# Input schema when browser_agent is used as a tool
class BrowserAgentInput(BaseModel):
    url: str

# Output schema for browser_agent
class BrowserAgentOutput(BaseModel):
    extracted_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None # If browser-use returns a message

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def browser_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = context._invocation_context
    
    # This state variable is set by this agent's own after_tool_callback
    # It contains the raw output from ExtractGitHubIssueDetailsTool
    raw_tool_output_from_state = None
    if invocation_ctx and invocation_ctx.session:
        raw_tool_output_from_state = invocation_ctx.session.state.get("temp:browser_raw_tool_output")

    if raw_tool_output_from_state:
        logger.info("BrowserAgent (instruction_provider): Raw tool output found in state. Instructing LLM to format it as per output_schema.")
        # Ensure it's a dict, as expected from ExtractGitHubIssueDetailsTool
        if not isinstance(raw_tool_output_from_state, dict):
            logger.error(f"BrowserAgent (instruction_provider): Data in 'temp:browser_raw_tool_output' is not a dict: {type(raw_tool_output_from_state)}")
            output_model = BrowserAgentOutput(error="Internal error: browser tool output was not a dictionary.")
        else:
            output_model = BrowserAgentOutput(
                extracted_details=raw_tool_output_from_state.get("extracted_details"),
                error=raw_tool_output_from_state.get("error"),
                message=raw_tool_output_from_state.get("message") # browser-use might return a message
            )
        
        json_to_output_llm = output_model.model_dump_json(exclude_none=True)
        
        if invocation_ctx and invocation_ctx.session: # Consume state
            invocation_ctx.session.state.pop("temp:browser_raw_tool_output", None)
            
        return f"Your final response for this turn MUST be exactly: '{json_to_output_llm}'"
    else:
        # Initial call: instruct LLM to call the ExtractGitHubIssueDetailsTool
        user_input_json_str = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
        url_to_fetch = ""
        if user_input_json_str:
            try:
                args_model = BrowserAgentInput.model_validate_json(user_input_json_str)
                url_to_fetch = args_model.url
            except Exception as e:
                logger.error(f"BrowserAgent (instruction_provider): Could not parse/validate input JSON: {user_input_json_str}. Error: {e}")
                # This will be the JSON output by this agent's LLM due to output_schema
                error_output = BrowserAgentOutput(error="Browser agent received invalid/malformed URL input.")
                return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"
        
        if not url_to_fetch:
            logger.error("BrowserAgent (instruction_provider): URL not provided/extracted from input.")
            error_output = BrowserAgentOutput(error="URL not provided to browser_agent.")
            return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"

        logger.info(f"BrowserAgent (instruction_provider): Instructing LLM to call '{ExtractGitHubIssueDetailsTool().name}' for URL: {url_to_fetch}")
        return f"Your ONLY task is to call the tool '{ExtractGitHubIssueDetailsTool().name}' with the argument 'url' set to '{url_to_fetch}'. Do not add any other text or commentary. A callback will process the tool's output."

def browser_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict  # Raw dict from ExtractGitHubIssueDetailsTool.run_async
) -> dict | None:    
    if tool.name == ExtractGitHubIssueDetailsTool().name:
        logger.info(f"BrowserAgent (after_tool_callback for {tool.name}): Received raw tool response: {str(tool_response)[:200]}")
        
        # Store the raw tool output in session state.
        # The instruction_provider will use this on the next turn to format the final LLM output.
        if isinstance(tool_response, dict):
            tool_context.state["temp:browser_raw_tool_output"] = tool_response
            logger.info("BrowserAgent (after_tool_callback): Stored raw tool response in 'temp:browser_raw_tool_output'.")
        else: # Should not happen if ExtractGitHubIssueDetailsTool works correctly
            error_payload = {"error": f"Tool '{tool.name}' provided non-dict output: {type(tool_response)}"}
            tool_context.state["temp:browser_raw_tool_output"] = error_payload
            logger.warning(f"BrowserAgent (after_tool_callback): Tool response was not a dict. Stored error payload. Type: {type(tool_response)}")

        tool_context.actions.skip_summarization = True # Critical: re-prompt browser_agent's instruction_provider
        return {"status": "tool_run_complete_data_in_state"} # Return a simple dict to satisfy ADK and trigger re-prompt
    return None

browser_agent = ADKAgent(
    name="browser_utility_agent",
    description="Fetches content from a URL and returns it as structured JSON.", # Added description
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=BrowserAgentInput,   # Schema for when this agent is CALLED as a tool
    output_schema=BrowserAgentOutput, # Schema for what this agent RETURNS when used as a tool
    instruction=browser_agent_instruction_provider, 
    tools=[ExtractGitHubIssueDetailsTool()], # Internal tool this agent uses
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=browser_agent_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    # Max output tokens should be enough for the JSON output_schema
    generate_content_config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20000) 
)