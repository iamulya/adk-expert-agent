# expert-agents/browser_agent.py
import logging
import json

from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

def browser_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict 
) -> genai_types.Content | None:
    if tool.name == ExtractGitHubIssueDetailsTool().name: 
        logger.info(f"BrowserAgent (after_tool_callback): Intercepted response from '{tool.name}'.")
        if isinstance(tool_response, dict) and "extracted_details" in tool_response:
            output_dict = {"extracted_details": tool_response["extracted_details"]}
            try:
                output_json_string = json.dumps(output_dict)
                logger.info(f"BrowserAgent (after_tool_callback): Returning direct JSON string: {output_json_string[:200]}...")
                tool_context.actions.skip_summarization = True # Make this Content final for browser_agent
                return genai_types.Content(parts=[genai_types.Part(text=output_json_string)])
            except TypeError as e:
                logger.error(f"BrowserAgent (after_tool_callback): Could not serialize to JSON: {e}. Returning raw.")
                error_json = json.dumps({"error": f"Failed to serialize browser output: {tool_response['extracted_details'][:100]}..."})
                tool_context.actions.skip_summarization = True
                return genai_types.Content(parts=[genai_types.Part(text=error_json)])
        else:
            logger.warning(f"BrowserAgent (after_tool_callback): '{tool.name}' did not return 'extracted_details'. Response: {tool_response}")
            error_json = json.dumps({"error": f"Tool '{tool.name}' did not provide 'extracted_details'."})
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_json)])
    return None

browser_agent = ADKAgent(
    name="browser_utility_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction="You are a utility agent. Your ONLY task is to execute the provided browser tool with the given URL. The tool's direct JSON output will be handled by a callback.",
    tools=[ExtractGitHubIssueDetailsTool()],
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=browser_agent_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20000) # LLM only makes tool call
)