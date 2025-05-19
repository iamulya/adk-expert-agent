# expert-agents/github_issue_fetcher_agent.py
import logging
import json
from pydantic import BaseModel, Field

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from .browser_agent import browser_agent # browser_agent itself is an ADKAgent
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

class GitHubIssueFetcherToolInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

# NO after_tool_callback in github_issue_fetcher_agent anymore.
# The output of its browser_agent tool call will be passed raw to root_agent.

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
                input_data = GitHubIssueFetcherToolInput.model_validate_json(first_part_text)
                issue_number_str = input_data.issue_number
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider): Could not parse issue_number from input '{first_part_text}': {e}")
                # This agent now directly returns errors as its final text output if input is bad.
                # AgentTool will pick this up.
                return "Your final response must be the JSON: {\"error\": \"Fetcher agent received invalid input. Expected an issue number.\"}"
    
    if issue_number_str:
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Instructing LLM to call browser_agent for URL: {github_url}")
        # This instruction tells the FetcherAgent's LLM to make a tool call.
        # The output of this tool call (browser_agent) will be the final output of this agent.
        return f"Your ONLY task is to call the '{browser_agent.name}' tool with the URL '{github_url}'. The output of this tool will be your direct response. Do not add any other text or commentary."
    else:
        logger.error("FetcherAgent (instruction_provider): No issue_number provided. This is an internal error.")
        return "Your final response must be the JSON: {\"error\": \"Fetcher agent was called without an issue number.\"}"

github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number, fetches its content using a browser, and directly returns the browser's output.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) # This is browser_utility_agent
    ],
    input_schema=GitHubIssueFetcherToolInput,
    before_model_callback=log_prompt_before_model_call,
    # after_tool_callback REMOVED - Let AgentTool pass browser_agent's output directly
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000, # Only needs to make a tool call
        top_p=0.6
    )
)