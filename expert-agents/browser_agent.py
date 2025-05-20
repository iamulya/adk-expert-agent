# expert-agents/browser_agent.py
import logging
import json
from pydantic import BaseModel # Add this

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

# Define the input schema for browser_agent when used as a tool
class BrowserAgentInput(BaseModel):
    url: str

# Helper function
def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def browser_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context
    
    user_input_json_str = "" # Expect '{"url": "..."}'
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        user_input_json_str = get_text_from_content(invocation_ctx.user_content)

    browser_tool_output_data = None
    if invocation_ctx and hasattr(invocation_ctx, 'session') and invocation_ctx.session:
        browser_tool_output_data = invocation_ctx.session.state.get("temp:browser_tool_output_data")

    if browser_tool_output_data:
        if hasattr(invocation_ctx, 'session'): 
            invocation_ctx.session.state.pop("temp:browser_tool_output_data", None)
        
        logger.info(f"BrowserAgent (instruction_provider): Tool output data found in state. Instructing LLM to output sentinel string 'BROWSER_DATA_READY'.")
        return "Your final response for this turn MUST be exactly: 'BROWSER_DATA_READY'"
    else:
        url_to_fetch = ""
        if user_input_json_str:
            try:
                # user_input_json_str is now expected to be '{"url": "..."}' due to input_schema
                args_dict = json.loads(user_input_json_str)
                url_to_fetch = args_dict.get("url")
                if not url_to_fetch: # Ensure 'url' key exists and has a value
                    raise ValueError("'url' key missing or empty in parsed JSON.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"BrowserAgent (instruction_provider): Could not parse/validate input JSON to get URL: {user_input_json_str}. Error: {e}")
                return "Your final response must be the JSON: {\"error\": \"Browser agent received invalid or malformed URL input.\"}"
        
        if not url_to_fetch:
            logger.error("BrowserAgent (instruction_provider): URL not provided or extracted from input.")
            # This case should ideally be caught by the try-except above if user_input_json_str was present but malformed.
            # If user_input_json_str was empty, it implies an issue with how AgentTool passed args.
            return "Your final response must be the JSON: {\"error\": \"URL not provided to browser_agent input.\"}"

        logger.info(f"BrowserAgent (instruction_provider): Instructing LLM to call tool '{ExtractGitHubIssueDetailsTool().name}' for URL: {url_to_fetch}")
        return f"Your ONLY task is to call the tool '{ExtractGitHubIssueDetailsTool().name}' with the argument 'url' set to '{url_to_fetch}'. Do not add any other text or commentary."

def browser_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict 
) -> dict | None:    
    if tool.name == ExtractGitHubIssueDetailsTool().name:
        logger.info(f"BrowserAgent (after_tool_callback): Intercepted response from '{tool.name}'. Raw response: {str(tool_response)[:200]}")
        
        if isinstance(tool_response, dict) and ("extracted_details" in tool_response or "error" in tool_response):
            tool_context.state["temp:browser_tool_output_data"] = tool_response
            logger.info(f"BrowserAgent (after_tool_callback): Stored tool response in session state 'temp:browser_tool_output_data'.")
            tool_context.actions.skip_summarization = True
            return tool_response 
        else:
            logger.warning(f"BrowserAgent (after_tool_callback): Tool '{tool.name}' did not return 'extracted_details' or 'error'. Response: {tool_response}")
            error_payload = {"error": f"Tool '{tool.name}' did not provide 'extracted_details' or 'error' in its response dict."}
            tool_context.state["temp:browser_tool_output_data"] = error_payload
            tool_context.actions.skip_summarization = True
            return error_payload
    return None

browser_agent = ADKAgent(
    name="browser_utility_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=BrowserAgentInput,  # Crucial addition
    instruction=browser_agent_instruction_provider, 
    tools=[ExtractGitHubIssueDetailsTool()],
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=browser_agent_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20000) # Reduced tokens as it only outputs a sentinel.
)