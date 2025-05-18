from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool 
from .browser_agent import browser_agent 
from .tools import get_gemini_api_key_from_secret_manager

# Import the raw context loader from root_agent.py
# This assumes root_agent.py defines load_raw_adk_context_for_root_agent
# and it's suitable for reuse, or create a specific one here.

from .context_loader import get_escaped_adk_context_for_llm, ADK_CONTEXT_DATA_FILE

get_gemini_api_key_from_secret_manager()

# RAW_ADK_FILE_CONTENT_FOR_GITHUB_AGENT and load_raw_adk_context_for_github_agent are removed

def github_issue_agent_instruction_provider(context: ReadonlyContext) -> str:
    # Get the prepared (escaped) ADK context from the shared loader
    adk_context_for_llm_with_note = get_escaped_adk_context_for_llm()
    
    instruction_template_simplified = """
You are a specialized agent for solving GitHub issues for 'google/adk-python', using Google's ADK version 0.5.0.
Your knowledge base includes the following ADK context:
--- START OF ADK CONTEXT ---
{ADK_CONTEXT_PLACEHOLDER}
--- END OF ADK CONTEXT ---

Your task is to handle a GitHub issue.
1.  Examine the user's request. If an issue number for 'google/adk-python' is mentioned, make a note of it.
2.  If you do not have an issue number, YOU MUST ask the user for "the GitHub issue number". This should be your only response for this turn.
3.  If you have the issue number (either from the current query or a previous turn):
    a. Construct the URL: `https://github.com/google/adk-python/issues/THE_ISSUE_NUMBER_YOU_HAVE`.
    b. Call the 'browser_utility_agent' tool with this exact URL.
4.  After the 'browser_utility_agent' tool returns details, use these details and your ADK knowledge to provide a solution or answer. This is your final response for this query.
"""
    # The .replace('%', '%%') is for Python's .format()
    return instruction_template_simplified.format(
        ADK_CONTEXT_PLACEHOLDER=adk_context_for_llm_with_note.replace('%', '%%')
    )

github_issue_solver_agent = ADKAgent(
    name="github_issue_solver_agent",
    model=Gemini(model="gemini-2.5-pro-preview-05-06"),
    instruction=github_issue_agent_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) 
    ],
    disallow_transfer_to_peers=True, 
)