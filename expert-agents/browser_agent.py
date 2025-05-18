from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from .tools import ExtractGitHubIssueDetailsTool, get_gemini_api_key_from_secret_manager

get_gemini_api_key_from_secret_manager()

browser_agent = ADKAgent(
    name="browser_utility_agent",
    model=Gemini(model="gemini-2.5-pro-preview-05-06"),
    instruction="You are a utility agent that uses a browser tool to fetch content from a given URL. Focus on executing the tool call based on the provided URL.",
    tools=[ExtractGitHubIssueDetailsTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)