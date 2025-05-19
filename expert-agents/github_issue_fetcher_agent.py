import re
import logging
from pydantic import BaseModel

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext # For after_tool_callback
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool # For type hinting in callback
from google.adk.tools.tool_context import ToolContext # For type hinting in callback
from google.genai import types as genai_types

from .browser_agent import browser_agent
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

BOILERPLATE_STRINGS_TO_REMOVE = [
    "**Is your feature request related to a problem? Please describe.**",
    "**Describe the solution you'd like**",
    "**Describe alternatives you've considered**",
    "**Describe the bug**",
    "**Minimal Reproduction**",
    "**Minimal steps to reproduce**",
    "**Desktop (please complete the following information):**",
    "** Please make sure you read the contribution guide and file the issues in the rigth place.**",
    "**To Reproduce**",
    "**Expected behavior**",
    "**Screenshots**",
    "**Additional context**"
]

def clean_github_issue_text(text: str) -> str:
    if not text:
        return ""
    cleaned_text = text
    for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(boilerplate, "")
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    return cleaned_text

class GitHubIssueFetcherInput(BaseModel):
    user_query: str

def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict, # Arguments passed to the tool
    tool_context: ToolContext, # Context for the tool that was just run
    tool_response: dict # The raw response from the tool
) -> genai_types.Content | None:
    """
    This callback runs after a tool (specifically browser_agent) is called by github_issue_fetcher_agent.
    It processes the browser_agent's output, cleans it, and returns the exact content
    that github_issue_fetcher_agent should output, bypassing its own LLM for this step.
    """
    # We are interested in the response from the browser_agent
    if tool.name == browser_agent.name:
        logger.info("GitHubIssueFetcherAgent (after_tool_callback): Processing browser_agent response.")
        raw_browser_tool_output = ""
        if isinstance(tool_response, dict) and "extracted_details" in tool_response:
            raw_browser_tool_output = tool_response["extracted_details"]

        if raw_browser_tool_output:
            cleaned_details = clean_github_issue_text(raw_browser_tool_output)
            if not cleaned_details:
                final_text = "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."
            else:
                final_text = f"---BEGIN CLEANED ISSUE TEXT---\n{cleaned_details}\n---END CLEANED ISSUE TEXT---"
        else: # Browser tool returned nothing or an error handled by the tool itself
            final_text = "The browser tool could not fetch or extract content from the GitHub issue. Please ensure the URL is correct or try again later."
        
        logger.info(f"GitHubIssueFetcherAgent (after_tool_callback): Returning direct content: {final_text[:100]}...")
        # Returning Content here will make it the direct output of the github_issue_fetcher_agent for this turn
        return genai_types.Content(parts=[genai_types.Part(text=final_text)])
    
    # For any other tool (if any were added), do nothing, let normal LLM processing continue
    return None


def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    user_query = ""
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context

    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content and \
       invocation_ctx.user_content.parts:
        first_part_text = invocation_ctx.user_content.parts[0].text
        if first_part_text:
            try:
                input_data = GitHubIssueFetcherInput.model_validate_json(first_part_text)
                user_query = input_data.user_query
            except Exception:
                user_query = first_part_text 
    
    logger.info(f"GitHubIssueFetcherAgent (instruction_provider): Received user_query: '{user_query}' for processing.")

    match = re.search(r"(?:google/adk-python.*(?:issue|ticket|bug|fix|feature)\s*(?:#)?(\d+)|(?:issue|ticket|bug|fix|feature)\s*(?:#)?(\d+).*google/adk-python)", user_query, re.IGNORECASE)
    issue_number = None
    if match:
        issue_number = match.group(1) or match.group(2)
        logger.info(f"GitHubIssueFetcherAgent (instruction_provider): Extracted issue_number: {issue_number} from query: '{user_query}'")

    # IMPORTANT: The logic to check for browser_agent output is now primarily handled
    # by the `after_tool_callback`. This instruction provider focuses on the initial step.
    
    if issue_number:
        logger.info(f"GitHubIssueFetcherAgent (instruction_provider): Issue number {issue_number} found. Instructing LLM to call browser_utility_agent.")
        github_url = f"https://github.com/google/adk-python/issues/{issue_number}"
        return f"You have identified issue number {issue_number}. Your task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. This is your only action for this turn. Do not add any conversational text."
    else:
        logger.info(f"GitHubIssueFetcherAgent (instruction_provider): No issue number found in query '{user_query}'. Instructing LLM to ask user.")
        # This output will be directly used by the root_agent
        return "Your final response MUST be exactly the following text, without any additions: Please provide the GitHub issue number for 'google/adk-python' that you would like me to look into."

github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Fetches and cleans details of a specific GitHub issue from 'google/adk-python'. It may ask for an issue number if not provided.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent)
    ],
    input_schema=GitHubIssueFetcherInput,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=github_issue_fetcher_after_tool_callback, # ADDED THIS
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=1024, # Reduced, as LLM only outputs tool call or "ask user"
        top_p=0.6
    )
)