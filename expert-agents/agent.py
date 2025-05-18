import os
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool

from .context_loader import get_escaped_adk_context_for_llm, ADK_CONTEXT_DATA_FILE
from .github_issue_agent import github_issue_solver_agent 
from .tools import get_gemini_api_key_from_secret_manager

load_dotenv()

def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    # Get the prepared (escaped) ADK context from the shared loader
    adk_context_for_llm_with_note = get_escaped_adk_context_for_llm()

    system_instruction_template = """
You are an expert on Google's Agent Development Kit (ADK) version 0.5.0.
Your primary role is to answer general questions about ADK or to delegate tasks to specialized agents.

When a user starts a conversation, greet them by introducing yourself as an ADK 0.5.0 expert.

If the user's query seems to be about solving a specific GitHub issue for 'google/adk-python' (e.g., they mention "issue", "GitHub", "bug report", "feature request" along with 'google/adk-python'):
1. Invoke the 'github_issue_solver_agent' tool to handle this. Do not ask for the issue number yourself; let the specialized agent manage that.

If the user's query is a general question about ADK (not a specific GitHub issue):
1. Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general questions):
--- START OF ADK CONTEXT ---
{ADK_CONTEXT_PLACEHOLDER}
--- END OF ADK CONTEXT ---
"""
    # The .replace('%', '%%') is for Python's .format() if any % are in the context.
    return system_instruction_template.format(
        # Still use ADK_CONTEXT_DATA_FILE for the filename in the prompt if needed
        filename_placeholder=ADK_CONTEXT_DATA_FILE.name, 
        ADK_CONTEXT_PLACEHOLDER=adk_context_for_llm_with_note.replace('%', '%%')
    )

API_KEY = get_gemini_api_key_from_secret_manager()

root_agent = ADKAgent(
    name="adk_expert_bot",
    model=Gemini(model="gemini-2.5-pro-preview-05-06"),
    instruction=root_agent_instruction_provider,
    tools=[
        AgentTool(agent=github_issue_solver_agent)
    ],
)