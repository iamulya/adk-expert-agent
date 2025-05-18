import os
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents import Agent as ADKAgent # Alias to avoid conflict
from google.adk.models import Gemini

# Import tools and helper
from .tools import get_gemini_api_key_from_secret_manager, ExtractGitHubIssueDetailsTool

# Load environment variables from .env file
load_dotenv()

# --- Global Variables ---
ADK_CONTEXT_DATA_FILE = Path(__file__).parent / "data" / "google-adk-python-0.5-with-test.txt"
ADK_FILE_CONTENT = ""

# --- Helper Functions ---
def load_adk_context_file_once():
    """Loads the ADK context file content once."""
    global ADK_FILE_CONTENT
    if not ADK_FILE_CONTENT:
        try:
            with open(ADK_CONTEXT_DATA_FILE, "r", encoding="utf-8") as f:
                ADK_FILE_CONTENT = f.read()
            print(f"Successfully loaded ADK context from {ADK_CONTEXT_DATA_FILE}")
        except FileNotFoundError:
            print(f"Error: ADK context file not found at {ADK_CONTEXT_DATA_FILE}")
            # Fallback or raise an error if the file is critical
            ADK_FILE_CONTENT = "ADK context file not found. Cannot provide detailed ADK information."
        except Exception as e:
            print(f"Error reading ADK context file: {e}")
            ADK_FILE_CONTENT = f"Error reading ADK context file: {e}. Cannot provide detailed ADK information."
    return ADK_FILE_CONTENT

# --- Agent Definition ---

# Ensure API key is loaded before Agent instantiation
API_KEY = get_gemini_api_key_from_secret_manager()
ADK_CONTEXT = load_adk_context_file_once()

# Define the system instruction for the main ADK agent
SYSTEM_INSTRUCTION = f"""
You are an expert on Google's Agent Development Kit (ADK) version 0.5.0.
Your knowledge base includes the following content from '{ADK_CONTEXT_DATA_FILE.name}':
--- START OF ADK CONTEXT ---
{ADK_CONTEXT}
--- END OF ADK CONTEXT ---

When a user starts a conversation, greet them by introducing yourself as an ADK 0.5.0 expert.

If the user's query is about solving a GitHub issue related to 'google/adk-python':
1. Check if the user has provided an issue number.
2. If the issue number is not provided, YOU MUST ask the user for the GitHub issue number. Do not attempt to guess or proceed without it.
3. Once you have the issue number, construct the full GitHub issue URL in the format: https://github.com/google/adk-python/issues/{{issue_number}}.
4. Call the 'extract_github_issue_details_tool' with this URL. This tool will use a browser agent to fetch the issue's title, body, and all comments.
5. After receiving the extracted details from the tool, use these details along with your ADK knowledge (from the context above) to formulate a comprehensive solution, provide insights, or answer questions about the issue. This is your final answer.

If the user's query is NOT about a GitHub issue:
1. Use your ADK knowledge (from the ADK context provided above) to answer the user's query directly. This is your final answer.
"""

root_agent = ADKAgent(
    name="adk_expert_bot",
    model=Gemini(model="gemini-2.5-pro-preview-05-06"), # ADK uses its own Gemini model class
    instruction=SYSTEM_INSTRUCTION,
    tools=[ExtractGitHubIssueDetailsTool()],
    # Enable function calling for the agent to use the tool
    # For Gemini, this is often implicit if tools are provided, but explicit configuration might be needed
    # depending on ADK version and specific Gemini model features.
    # ADK's LlmAgent handles tool configuration automatically.
)

# To run this agent with 'adk web .':
# 1. Make sure this file (agent.py) is in the root of your agent directory.
# 2. Ensure __init__.py exists in the same directory.
# 3. The 'adk web' command will automatically look for 'root_agent'.
