import re
import logging
from pydantic import BaseModel

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext # Stays as ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
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

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    user_query = ""
    
    # CORRECTED: Access user_content via _invocation_context
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
                user_query = first_part_text # Fallback
    
    logger.info(f"GitHubIssueFetcherAgent: Received user_query: '{user_query}' for processing.")

    match = re.search(r"(?:google/adk-python.*(?:issue|ticket|bug|fix|feature)\s*(?:#)?(\d+)|(?:issue|ticket|bug|fix|feature)\s*(?:#)?(\d+).*google/adk-python)", user_query, re.IGNORECASE)
    issue_number = None
    if match:
        issue_number = match.group(1) or match.group(2)
        logger.info(f"GitHubIssueFetcherAgent: Extracted issue_number: {issue_number} from query: '{user_query}'")

    raw_browser_tool_output = ""
    # The context passed here is ReadonlyContext, so we access _invocation_context
    # to get to session events.
    if invocation_ctx and hasattr(invocation_ctx, 'session') and \
       invocation_ctx.session and invocation_ctx.session.events:
        for event in reversed(invocation_ctx.session.events):
            if event.author == github_issue_fetcher_agent.name and event.get_function_responses(): # Check if it's a response to this agent's tool call
                for func_response in event.get_function_responses():
                    if func_response.name == browser_agent.name:
                        if isinstance(func_response.response, dict) and "extracted_details" in func_response.response:
                            raw_browser_tool_output = func_response.response["extracted_details"]
                            logger.info("GitHubIssueFetcherAgent: Found browser_agent output in context.")
                            break
                if raw_browser_tool_output:
                    break
    
    if raw_browser_tool_output:
        logger.info("GitHubIssueFetcherAgent: Processing browser output.")
        cleaned_details = clean_github_issue_text(raw_browser_tool_output)
        if not cleaned_details:
            return "The fetched GitHub issue content appears to be empty or contained only template text. No specific details to analyze."
        return f"You have successfully fetched and cleaned the GitHub issue content. Your final response MUST be exactly the following text, without any additional conversational fluff or explanations:\n\n---BEGIN CLEANED ISSUE TEXT---\n{cleaned_details}\n---END CLEANED ISSUE TEXT---"

    if issue_number:
        logger.info(f"GitHubIssueFetcherAgent: Issue number {issue_number} found. Calling browser_utility_agent.")
        github_url = f"https://github.com/google/adk-python/issues/{issue_number}"
        return f"You have identified issue number {issue_number}. Your task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. This is your only action for this turn. Do not add any conversational text."
    else:
        logger.info(f"GitHubIssueFetcherAgent: No issue number found in query '{user_query}'. Asking user.")
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
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000,
        top_p=0.6
    )
)