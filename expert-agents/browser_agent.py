from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

get_gemini_api_key_from_secret_manager()

browser_agent = ADKAgent(
    name="browser_utility_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction="You are a utility agent that uses a browser tool to fetch content from a given URL. Focus on executing the tool call based on the provided URL.",
    tools=[ExtractGitHubIssueDetailsTool()],
    before_model_callback=log_prompt_before_model_call,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)