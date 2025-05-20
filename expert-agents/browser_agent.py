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
    message: Optional[str] = None 

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def browser_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = context._invocation_context
    
    raw_tool_output_from_state = None
    if invocation_ctx and invocation_ctx.session:
        raw_tool_output_from_state = invocation_ctx.session.state.get("temp:browser_raw_tool_output")

    if raw_tool_output_from_state:
        logger.info("BrowserAgent (instruction_provider): Raw tool output found in state. Instructing LLM to format it as per output_schema.")
        
        output_data_for_llm = {}
        if isinstance(raw_tool_output_from_state, dict):
            output_data_for_llm["extracted_details"] = raw_tool_output_from_state.get("extracted_details")
            output_data_for_llm["error"] = raw_tool_output_from_state.get("error")
            output_data_for_llm["message"] = raw_tool_output_from_state.get("message")
        else:
            logger.error(f"BrowserAgent (instruction_provider): Data in 'temp:browser_raw_tool_output' is not a dict: {type(raw_tool_output_from_state)}")
            output_data_for_llm["error"] = "Internal error: browser tool output was not a dictionary."
        
        # Create the JSON string for the LLM to output, ensuring it matches BrowserAgentOutput
        # Pydantic will exclude None values by default if not specified otherwise in model_config
        output_model_instance = BrowserAgentOutput(**output_data_for_llm)
        json_to_output_llm = output_model_instance.model_dump_json(exclude_none=True)
        
        if invocation_ctx and invocation_ctx.session: 
            invocation_ctx.session.state.pop("temp:browser_raw_tool_output", None)
            
        return f"Your final response for this turn MUST be exactly: '{json_to_output_llm}'"
    else:
        user_input_json_str = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
        url_to_fetch = ""
        if user_input_json_str:
            try:
                args_model = BrowserAgentInput.model_validate_json(user_input_json_str)
                url_to_fetch = args_model.url
            except Exception as e:
                logger.error(f"BrowserAgent (instruction_provider): Could not parse/validate input JSON: {user_input_json_str}. Error: {e}")
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
    tool_response: dict 
) -> dict | None:    
    if tool.name == ExtractGitHubIssueDetailsTool().name:
        logger.info(f"BrowserAgent (after_tool_callback for {tool.name}): Received raw tool response: {str(tool_response)[:200]}")
        
        if isinstance(tool_response, dict):
            tool_context.state["temp:browser_raw_tool_output"] = tool_response
            logger.info("BrowserAgent (after_tool_callback): Stored raw tool response in 'temp:browser_raw_tool_output'.")
        else: 
            error_payload = {"error": f"Tool '{tool.name}' provided non-dict output: {type(tool_response)}"}
            tool_context.state["temp:browser_raw_tool_output"] = error_payload
            logger.warning(f"BrowserAgent (after_tool_callback): Tool response was not a dict. Stored error payload. Type: {type(tool_response)}")

        tool_context.actions.skip_summarization = True 
        return {"status": "tool_run_complete_data_in_state"} 
    return None

browser_agent = ADKAgent(
    name="browser_utility_agent",
    description="Fetches content from a URL and returns it as structured JSON including extracted_details, error, or message.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=BrowserAgentInput,  
    output_schema=BrowserAgentOutput, 
    instruction=browser_agent_instruction_provider, 
    tools=[ExtractGitHubIssueDetailsTool()], 
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=browser_agent_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20000) 
)