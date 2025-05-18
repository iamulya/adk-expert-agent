import os
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool

from google.genai import types as genai_types

from .context_loader import get_escaped_adk_context_for_llm, ADK_CONTEXT_DATA_FILE
from .github_issue_agent import github_issue_solver_agent 
from .tools import get_gemini_api_key_from_secret_manager

load_dotenv()

def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm_with_note = get_escaped_adk_context_for_llm()

    # This instruction will be used by the root_agent's LLM for *every* turn.
    # The LLM will see the conversation history, including any FunctionResponse from tools.
    system_instruction_template = """
You are an expert on Google's Agent Development Kit (ADK) version 0.5.0.
Your primary role is to answer general questions about ADK or to delegate tasks to specialized agents.

When a user starts a conversation, greet them by introducing yourself as an ADK 0.5.0 expert.

Consider the user's latest query:
- If the query seems to be about solving a specific GitHub issue for 'google/adk-python' (e.g., they mention "issue", "GitHub", "bug report", "feature request" along with 'google/adk-python'):
    1. Your primary action is to call the '{github_agent_name}' tool to handle this.
    2. Do not try to ask for the issue number yourself; let the specialized agent manage that.
    3. After calling the '{github_agent_name}' tool, it will return detailed information as a tool response.
    4. **Your final step is to take the complete, unabbreviated information returned by the '{github_agent_name}' tool and present it directly to the user as your answer.** Do not summarize it or add significant conversational fluff beyond a polite framing if necessary. This is your final response for the GitHub issue query.

- If the user's query is a general question about ADK (and not a specific GitHub issue you just processed with a tool):
    1. Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general ADK questions):
--- START OF ADK CONTEXT ---
{{ADK_CONTEXT_PLACEHOLDER}}
--- END OF ADK CONTEXT ---
"""
    # Using .replace for ADK_CONTEXT_PLACEHOLDER and Python f-string for github_agent_name
    # This ensures that the main template is formatted first, then the large context block is inserted.
    formatted_template = system_instruction_template.format(
        github_agent_name=github_issue_solver_agent.name # Get the actual agent name
    )
    
    # Now replace the ADK_CONTEXT_PLACEHOLDER.
    # The .replace('%', '%%') is for Python's .format() if any % are in the context,
    # though since we are using .replace here for the ADK context, it's less critical
    # for this specific placeholder but good practice if the context string was complex.
    final_instruction = formatted_template.replace(
        "{ADK_CONTEXT_PLACEHOLDER}",
        adk_context_for_llm_with_note.replace('%', '%%')
    )
    return final_instruction

API_KEY = get_gemini_api_key_from_secret_manager()

root_agent = ADKAgent(
    name="adk_expert_bot",
    model=Gemini(model="gemini-2.5-pro-preview-05-06"),
    instruction=root_agent_instruction_provider,
    tools=[
        AgentTool(agent=github_issue_solver_agent)
    ],
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=65536,
        top_p=0.6,
    )
)