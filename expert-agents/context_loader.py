"""
This module handles the loading and preparation of the ADK knowledge context.

The primary knowledge base for the agent is a text file containing information
about the ADK. This module provides functions to load this file and process its
content for safe inclusion in LLM prompts.
"""

from pathlib import Path

# Define constants and globals related to ADK context loading here
# Path to the text file containing the ADK knowledge base.
ADK_CONTEXT_DATA_FILE = Path(__file__).parent / "data" / "google-adk-python-1.2.0.txt"
# Global variable to cache the raw file content, avoiding repeated file reads.
RAW_ADK_FILE_CONTENT = ""  # This can be a single shared raw content


def load_raw_adk_context_file_once():
    """
    Loads the content of the ADK context file into a global variable.

    This function reads the file only once and caches the content.
    If the file cannot be read, it populates the cache with an error message.

    Returns:
        The raw string content of the ADK context file or an error message.
    """
    global RAW_ADK_FILE_CONTENT  # Use the shared global
    if not RAW_ADK_FILE_CONTENT:
        try:
            with open(ADK_CONTEXT_DATA_FILE, "r", encoding="utf-8") as f:
                RAW_ADK_FILE_CONTENT = f.read()
            print(f"Successfully loaded RAW ADK context from {ADK_CONTEXT_DATA_FILE}")
        except FileNotFoundError:
            RAW_ADK_FILE_CONTENT = (
                "ADK context file not found. Cannot provide detailed ADK information."
            )
            print(f"Error: {RAW_ADK_FILE_CONTENT}")
        except Exception as e:
            RAW_ADK_FILE_CONTENT = f"Error reading ADK context file: {e}. Cannot provide detailed ADK information."
            print(f"Error: {RAW_ADK_FILE_CONTENT}")
    return RAW_ADK_FILE_CONTENT


def get_escaped_adk_context_for_llm() -> str:
    """
    Prepares the ADK context for safe inclusion in an LLM prompt.

    LLMs can interpret curly braces `{}` as placeholders for templating,
    which can cause issues when the context itself contains code examples
    with curly braces. This function replaces them with textual representations
    and adds a note to the LLM on how to interpret them.

    Returns:
        The processed and escaped ADK context string, ready for the LLM.
    """
    raw_adk_context = load_raw_adk_context_file_once()

    # Replace curly braces to prevent them from being interpreted as prompt template variables.
    adk_context_textually_escaped = raw_adk_context.replace(
        "{", "<curly_brace_open>"
    ).replace("}", "<curly_brace_close>")

    # Add a note to the LLM to ensure it understands the escaped braces correctly.
    interpretation_note = "\n[LLM Note: In the ADK context below, '<curly_brace_open>' represents '{' and '<curly_brace_close>' represents '}'. Please interpret them as such when understanding code examples.]\n"

    return interpretation_note + adk_context_textually_escaped
