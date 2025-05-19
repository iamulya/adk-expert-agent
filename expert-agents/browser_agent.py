# expert-agents/browser_agent.py
import logging # Add logging
import json # For creating the JSON string output

from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.base_tool import BaseTool # For type hinting
from google.adk.tools.tool_context import ToolContext # For type hinting
from google.genai import types as genai_types # For genai_types.Content

from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__) # Add logger
get_gemini_api_key_from_secret_manager()

def browser_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict # This is the direct output from ExtractGitHubIssueDetailsTool
) -> genai_types.Content | None:
    """
    This callback runs after ExtractGitHubIssueDetailsTool is called by browser_agent.
    It ensures that the raw 'extracted_details' are passed through without summarization.
    """
    if tool.name == ExtractGitHubIssueDetailsTool().name: # Or check against a fixed name string
        logger.info(f"BrowserAgent (after_tool_callback): Intercepted response from '{tool.name}'.")
        if isinstance(tool_response, dict) and "extracted_details" in tool_response:
            # We want browser_agent's final output to be the JSON string representation
            # of what ExtractGitHubIssueDetailsTool returned.
            # This makes it easier for the calling agent (github_issue_fetcher_agent) to parse.
            output_dict = {"extracted_details": tool_response["extracted_details"]}
            try:
                output_json_string = json.dumps(output_dict)
                logger.info(f"BrowserAgent (after_tool_callback): Returning direct JSON string: {output_json_string[:200]}...")
                return genai_types.Content(parts=[genai_types.Part(text=output_json_string)])
            except TypeError as e:
                logger.error(f"BrowserAgent (after_tool_callback): Could not serialize tool_response to JSON: {e}. Returning raw details.")
                # Fallback: return just the string details if JSON serialization fails (should be rare)
                return genai_types.Content(parts=[genai_types.Part(text=tool_response["extracted_details"])])
        else:
            logger.warning(f"BrowserAgent (after_tool_callback): '{tool.name}' did not return 'extracted_details'. Response: {tool_response}")
            # Return an error message or the raw response as text
            error_text = f"Error: Tool '{tool.name}' did not provide 'extracted_details'. Output was: {str(tool_response)}"
            return genai_types.Content(parts=[genai_types.Part(text=error_text)])
    return None


browser_agent = ADKAgent(
    name="browser_utility_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    # The instruction can be simplified as the callback will now primarily handle the output shaping
    instruction="You are a utility agent that uses a browser tool to fetch content from a given URL. Your primary job is to execute the tool.",
    tools=[ExtractGitHubIssueDetailsTool()],
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=browser_agent_after_tool_callback, # ADDED THIS
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    # max_output_tokens can be small if the LLM is only making a tool call
    generate_content_config=genai_types.GenerateContentConfig
    (
        temperature=0,
        max_output_tokens=20000
    )
)