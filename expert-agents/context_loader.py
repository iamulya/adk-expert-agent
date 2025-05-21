# expert-agents/context_loader.py
from pathlib import Path

# Define constants and globals related to ADK context loading here
ADK_CONTEXT_DATA_FILE = Path(__file__).parent / "data" / "google-adk-python-1.0.txt"
RAW_ADK_FILE_CONTENT = "" # This can be a single shared raw content

def load_raw_adk_context_file_once():
    """Loads the ADK context file content once, WITHOUT escaping."""
    global RAW_ADK_FILE_CONTENT # Use the shared global
    if not RAW_ADK_FILE_CONTENT:
        try:
            with open(ADK_CONTEXT_DATA_FILE, "r", encoding="utf-8") as f:
                RAW_ADK_FILE_CONTENT = f.read()
            print(f"Successfully loaded RAW ADK context from {ADK_CONTEXT_DATA_FILE}")
        except FileNotFoundError:
            RAW_ADK_FILE_CONTENT = "ADK context file not found. Cannot provide detailed ADK information."
            print(f"Error: {RAW_ADK_FILE_CONTENT}")
        except Exception as e:
            RAW_ADK_FILE_CONTENT = f"Error reading ADK context file: {e}. Cannot provide detailed ADK information."
            print(f"Error: {RAW_ADK_FILE_CONTENT}")
    return RAW_ADK_FILE_CONTENT

def get_escaped_adk_context_for_llm() -> str:
    """
    Loads the raw ADK context (if not already loaded) and returns
    a version with curly braces textually replaced for safe LLM prompting,
    along with an interpretation note.
    """
    raw_adk_context = load_raw_adk_context_file_once()
    
    adk_context_textually_escaped = raw_adk_context.replace('{', '<curly_brace_open>') \
                                                   .replace('}', '<curly_brace_close>')
    
    interpretation_note = "\n[LLM Note: In the ADK context below, '<curly_brace_open>' represents '{' and '<curly_brace_close>' represents '}'. Please interpret them as such when understanding code examples.]\n"
    
    return interpretation_note + adk_context_textually_escaped