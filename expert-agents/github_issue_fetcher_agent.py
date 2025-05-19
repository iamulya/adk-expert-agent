import re
import logging
from pydantic import BaseModel

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from .browser_agent import browser_agent
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

logger = logging.getLogger(__name__)

# Ensure API key is loaded when this module is imported
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

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    # This agent is called as a tool by the root_agent.
    # The root_agent will pass the user's query.
    # We expect user_query to be in context.user_content
    
    user_query = ""
    if context.user_content and context.user_content.parts:
        # Assuming the input from root_agent is a simple text part
        # or a JSON string of GitHubIssueFetcherInput
        first_part_text = context.user_content.parts[0].text
        if first_part_text:
            try:
                # Try to parse as JSON if root_agent sends structured input
                input_data = GitHubIssueFetcherInput.model_validate_json(first_part_text)
                user_query = input_data.user_query
            except Exception:
                # Fallback to treating it as plain text
                user_query = first_part_text
    
    logger.info(f"GitHubIssueFetcherAgent: Received user_query: '{user_query}'")

    # Regex to find issue numbers like 'issue 123', 'issue #123', 'ticket 123'
    # It looks for 'google/adk-python' and then an issue number.
    match = re.search(r"(?:google/adk-python.*(?:issue|ticket|bug|fix|feature)\s*(?:#)?(\d+)|(?:issue|ticket|bug|fix|feature)\s*(?:#)?(\d+).*google/adk-python)", user_query, re.IGNORECASE)
    issue_number = None
    if match:
        # Check both capturing groups
        issue_number = match.group(1) or match.group(2)
        logger.info(f"GitHubIssueFetcherAgent: Extracted issue_number: {issue_number}")

    # Check for browser_agent's output in the context
    # This means the browser_agent tool was called in a previous turn of *this* agent
    raw_browser_tool_output = ""
    if hasattr(context, '_invocation_context') and \
       hasattr(context._invocation_context, 'session') and \
       context._invocation_context.session and \
       context._invocation_context.session.events:
        # Look for the last function response from browser_agent
        for event in reversed(context._invocation_context.session.events):
            if event.get_function_responses():
                for func_response in event.get_function_responses():
                    # Ensure the response is from browser_agent and meant for this fetcher agent
                    if func_response.name == browser_agent.name:
                         # and event.author == github_issue_fetcher_agent.name: # This check might be too restrictive
                        if isinstance(func_response.response, dict) and "extracted_details" in func_response.response:
                            raw_browser_tool_output = func_response.response["extracted_details"]
                            logger.info("GitHubIssueFetcherAgent: Found browser_agent output in context.")
                            break
                if raw_browser_tool_output:
                    break
    
    if raw_browser_tool_output:
        # Phase 2: Browser output received, clean it and return
        logger.info("GitHubIssueFetcherAgent: Processing browser output.")
        cleaned_details = clean_github_issue_text(raw_browser_tool_output)
        if not cleaned_details:
            return "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."
        # This agent's job is to return the cleaned text.
        # The LLM for this agent should directly output this.
        return f"You have successfully fetched and cleaned the GitHub issue content. Your final response MUST be exactly the following text, without any additional conversational fluff or explanations:\n\n---BEGIN CLEANED ISSUE TEXT---\n{cleaned_details}\n---END CLEANED ISSUE TEXT---"

    # Phase 1: No browser output yet
    if issue_number:
        logger.info(f"GitHubIssueFetcherAgent: Issue number {issue_number} found. Calling browser_utility_agent.")
        # Construct URL and call browser_utility_agent
        github_url = f"https://github.com/google/adk-python/issues/{issue_number}"
        # The instruction tells the LLM to call the tool.
        return f"You have identified issue number {issue_number}. Your task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. This is your only action for this turn. Do not add any conversational text."
    else:
        logger.info("GitHubIssueFetcherAgent: No issue number found. Asking user.")
        # Ask the root_agent (via tool response) to ask the user
        return "Your final response MUST be exactly the following text, without any additions: Please provide the GitHub issue number for 'google/adk-python' that you would like me to look into."

# This agent will be called as a tool by the root_agent.
# Its input can be the raw user query.
github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Fetches and cleans details of a specific GitHub issue from 'google/adk-python'. It may ask for an issue number if not provided.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) # This agent can call the browser_agent
    ],
    input_schema=GitHubIssueFetcherInput, # Root agent will provide user_query
    before_model_callback=log_prompt_before_model_call,
    disallow_transfer_to_parent=True, # It's a tool, should not transfer
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000, # Increased if cleaned text can be large
        top_p=0.6
    )
)