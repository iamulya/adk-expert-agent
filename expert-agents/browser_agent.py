# expert-agents/browser_agent.py
import logging
import json

from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext # Added for instruction provider
from google.genai import types as genai_types

from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

# Helper function (if not already available or imported from elsewhere)
def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def browser_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context
    
    user_input_json_str = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        user_input_json_str = get_text_from_content(invocation_ctx.user_content)

    browser_tool_output_data = None
    if invocation_ctx and hasattr(invocation_ctx, 'session') and invocation_ctx.session:
        # Check if temp:browser_tool_output_data exists in the current session state
        browser_tool_output_data = invocation_ctx.session.state.get("temp:browser_tool_output_data")

    if browser_tool_output_data:
        if hasattr(invocation_ctx, 'session'): # Ensure session exists before pop
            # Consume the stored result from state
            invocation_ctx.session.state.pop("temp:browser_tool_output_data", None)
        
        logger.info(f"BrowserAgent (instruction_provider): Tool output data found in state. Instructing LLM to output sentinel string 'BROWSER_DATA_READY'.")
        # This instruction makes the LLM output the sentinel string as its final text response
        return "Your final response for this turn MUST be exactly: 'BROWSER_DATA_READY'"
    else:
        # Initial call: instruct LLM to call the tool
        url_to_fetch = ""
        if user_input_json_str:
            try:
                # AgentTool passes arguments to the sub-agent as a JSON string of a dict.
                # For ExtractGitHubIssueDetailsTool, the argument key is "url".
                args_dict = json.loads(user_input_json_str)
                url_to_fetch = args_dict.get("url")
            except json.JSONDecodeError:
                logger.error(f"BrowserAgent (instruction_provider): Could not parse input JSON to get URL: {user_input_json_str}")
                # This agent's LLM will output this JSON string.
                return "Your final response must be the JSON: {\"error\": \"Browser agent received invalid input. URL malformed.\"}"
        
        if not url_to_fetch:
            logger.error("BrowserAgent (instruction_provider): URL not provided in input.")
            return "Your final response must be the JSON: {\"error\": \"URL not provided to browser_agent.\"}"

        logger.info(f"BrowserAgent (instruction_provider): Instructing LLM to call tool '{ExtractGitHubIssueDetailsTool().name}' for URL: {url_to_fetch}")
        # This instruction makes the LLM call the tool
        return f"Your ONLY task is to call the tool '{ExtractGitHubIssueDetailsTool().name}' with the argument 'url' set to '{url_to_fetch}'. Do not add any other text or commentary."

def browser_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict  # This is the raw dict output from ExtractGitHubIssueDetailsTool.run_async
) -> dict | None:     # MUST return dict or None as per ADK LlmAgent.after_tool_callback signature
    if tool.name == ExtractGitHubIssueDetailsTool().name:
        logger.info(f"BrowserAgent (after_tool_callback): Intercepted response from '{tool.name}'. Raw response: {str(tool_response)[:200]}")
        
        # tool_response is already the dict {"extracted_details": "..."} or {"error": "..."}
        if isinstance(tool_response, dict) and ("extracted_details" in tool_response or "error" in tool_response):
            # Store the tool's result in the session state so the instruction_provider can access it next turn.
            tool_context.state["temp:browser_tool_output_data"] = tool_response
            logger.info(f"BrowserAgent (after_tool_callback): Stored tool response in session state 'temp:browser_tool_output_data'.")
            # skip_summarization ensures the LLM doesn't try to summarize this tool_response dict.
            # The agent will re-prompt based on the new state via the instruction_provider.
            tool_context.actions.skip_summarization = True
            return tool_response # Return the dict; ADK will form a FunctionResponse part from this.
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
    instruction=browser_agent_instruction_provider, 
    tools=[ExtractGitHubIssueDetailsTool()],
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=browser_agent_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20000) # Max output tokens for flash model can be large if it was echoing JSON, but now it's just a sentinel. Still, keeping it high doesn't hurt for flash.
)