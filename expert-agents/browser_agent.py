# expert-agents/browser_agent.py
import logging
import json
from pydantic import BaseModel
from typing import Optional

from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
# BaseTool and ToolContext are not strictly needed here anymore for ExtractGitHubIssueDetailsTool
# but ExtractGitHubIssueDetailsTool itself uses them.
from google.adk.tools.base_tool import BaseTool 
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.invocation_context import InvocationContext # For creating dummy ToolContext
from google.genai import types as genai_types

from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

class BrowserAgentInput(BaseModel):
    url: str

class BrowserAgentOutput(BaseModel):
    extracted_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# This function now needs to be async because it will await run_async
async def browser_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = context._invocation_context
    
    user_input_json_str = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
    url_to_fetch = ""

    if not user_input_json_str:
        logger.error("BrowserAgent (instruction_provider): No input JSON string provided.")
        error_output = BrowserAgentOutput(error="Browser agent did not receive any input.")
        return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"

    try:
        args_model = BrowserAgentInput.model_validate_json(user_input_json_str)
        url_to_fetch = args_model.url
    except Exception as e:
        logger.error(f"BrowserAgent (instruction_provider): Could not parse/validate input JSON: {user_input_json_str}. Error: {e}")
        error_output = BrowserAgentOutput(error="Browser agent received invalid/malformed URL input.")
        return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"
    
    if not url_to_fetch:
        logger.error("BrowserAgent (instruction_provider): URL not extracted from input.")
        error_output = BrowserAgentOutput(error="URL not provided to browser_agent after parsing.")
        return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"

    logger.info(f"BrowserAgent (instruction_provider): Directly calling ExtractGitHubIssueDetailsTool for URL: {url_to_fetch}")
    
    # Directly instantiate and run the tool
    # We need a dummy ToolContext for the tool's run_async method.
    # The InvocationContext for this agent is `invocation_ctx`.
    dummy_tool_context_for_extraction_tool = ToolContext(invocation_ctx)
    extraction_tool = ExtractGitHubIssueDetailsTool()
    
    try:
        tool_result_dict = await extraction_tool.run_async(
            args={"url": url_to_fetch}, 
            tool_context=dummy_tool_context_for_extraction_tool
        )
    except Exception as e:
        logger.error(f"BrowserAgent (instruction_provider): Error running ExtractGitHubIssueDetailsTool directly: {e}", exc_info=True)
        tool_result_dict = {"error": f"Failed to execute browser tool: {e}"}

    # Prepare the output based on the tool's result
    output_data_for_llm = {}
    if isinstance(tool_result_dict, dict):
        output_data_for_llm["extracted_details"] = tool_result_dict.get("extracted_details")
        output_data_for_llm["error"] = tool_result_dict.get("error")
        output_data_for_llm["message"] = tool_result_dict.get("message")
    else:
        logger.error(f"BrowserAgent (instruction_provider): ExtractGitHubIssueDetailsTool returned non-dict: {type(tool_result_dict)}")
        output_data_for_llm["error"] = "Internal error: Browser tool helper returned unexpected data type."

    output_model_instance = BrowserAgentOutput(**output_data_for_llm)
    json_to_output_llm = output_model_instance.model_dump_json(exclude_none=True)
        
    return f"Your final response for this turn MUST be exactly: '{json_to_output_llm}'"


# after_tool_callback is no longer needed for ExtractGitHubIssueDetailsTool
# as it's not an ADK-managed tool for this agent.
# If browser_agent itself had other tools it called via LLM, this would handle them.
def browser_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict 
) -> dict | None:    
    logger.warning(f"BrowserAgent (after_tool_callback): Unexpected tool call intercepted: {tool.name}. This agent should not be calling ADK tools via LLM.")
    return None

browser_agent = ADKAgent(
    name="browser_utility_agent",
    description="Fetches content from a URL and returns it as structured JSON including extracted_details, error, or message.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=BrowserAgentInput,  
    output_schema=BrowserAgentOutput, 
    instruction=browser_agent_instruction_provider, 
    tools=[], # IMPORTANT: tools list is now empty
    before_model_callback=log_prompt_before_model_call,
    # after_tool_callback can be set to None or a no-op if no other tools are used by LLM
    after_tool_callback=None, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20000) 
)