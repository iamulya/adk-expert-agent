# expert-agents/github_issue_fetcher_agent.py
import re
import logging
import json # Ensure json is imported
from pydantic import BaseModel, Field

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext 
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

# ... BOILERPLATE_STRINGS_TO_REMOVE and clean_github_issue_text ...
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
    "Additional context",
    "Here is the content of the GitHub issue:",
    "The content of the GitHub issue is:"
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
    # This agent is now simpler: it receives an issue number and fetches.
    # The root_agent is responsible for extracting the issue number initially.
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")
    # Alternatively, we could pass the full URL if root_agent constructs it.
    # github_url: Optional[str] = Field(default=None, description="The full URL to the GitHub issue.")

def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: str | dict 
) -> genai_types.Content | None:
    if tool.name == browser_agent.name:
        logger.info(f"FetcherAgent (after_tool_callback): Processing response from '{tool.name}'. Raw type: {type(tool_response)}, value: {str(tool_response)[:300]}...")
        
        text_output_from_browser_agent_tool_wrapper = ""
        if isinstance(tool_response, str):
            text_output_from_browser_agent_tool_wrapper = tool_response
        elif isinstance(tool_response, dict) and "text" in tool_response:
            text_output_from_browser_agent_tool_wrapper = tool_response["text"]
        else:
            logger.warning(f"FetcherAgent (after_tool_callback): Unexpected response format from {tool.name}. Got: {tool_response}")
            final_text = "Error: Received unexpected response format from the browser utility."
            return genai_types.Content(parts=[genai_types.Part(text=final_text)])

        raw_extracted_details = ""
        try:
            parsed_output = json.loads(text_output_from_browser_agent_tool_wrapper)
            if isinstance(parsed_output, dict) and "extracted_details" in parsed_output:
                raw_extracted_details = parsed_output["extracted_details"]
            else: # Fallback if JSON structure is not as expected but still valid JSON
                raw_extracted_details = text_output_from_browser_agent_tool_wrapper
        except json.JSONDecodeError: # Not JSON, assume it's the direct text
            raw_extracted_details = text_output_from_browser_agent_tool_wrapper
        
        if raw_extracted_details and "Error:" not in raw_extracted_details : # Basic check for error propagation
            cleaned_details = clean_github_issue_text(raw_extracted_details)
            if not cleaned_details:
                final_text = "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."
            else:
                final_text = f"---BEGIN CLEANED ISSUE TEXT---\n{cleaned_details}\n---END CLEANED ISSUE TEXT---"
        else: 
            final_text = "The browser tool could not fetch or extract content from the GitHub issue. Please ensure the URL is correct or try again later."
            if raw_extracted_details and "Error:" in raw_extracted_details:
                 final_text = raw_extracted_details # Relay specific error

        logger.info(f"FetcherAgent (after_tool_callback): Returning direct content: {final_text[:100]}...")
        return genai_types.Content(parts=[genai_types.Part(text=final_text)])
    return None

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    issue_number_str = None
    
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context

    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content and \
       invocation_ctx.user_content.parts:
        first_part_text = invocation_ctx.user_content.parts[0].text
        if first_part_text:
            try:
                # Expecting input like {"issue_number": "123"} from root_agent
                input_data = GitHubIssueFetcherToolInput.model_validate_json(first_part_text)
                issue_number_str = input_data.issue_number
                logger.info(f"FetcherAgent (instruction_provider): Received issue_number: '{issue_number_str}'.")
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider): Could not parse issue_number from input '{first_part_text}': {e}")
                # This agent should ideally always receive an issue_number now.
                # If not, it's an error in how root_agent called it.
                return "Error: This agent expects an issue number to be provided by the calling agent."
    
    if issue_number_str:
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Instructing LLM to call browser_agent for URL: {github_url}")
        return f"Your task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. This is your only action for this turn. Do not add any conversational text."
    else:
        # This path should ideally not be reached if root_agent calls this tool correctly.
        logger.error("FetcherAgent (instruction_provider): No issue_number was provided or parsed. This indicates an issue with how this agent was called.")
        return "Error: No issue number was provided to fetch."

github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content using a browser, and returns the cleaned content.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent)
    ],
    input_schema=GitHubIssueFetcherToolInput, # Expects issue_number
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