from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool 

from google.genai import types as genai_types

from .browser_agent import browser_agent 
from .tools import get_gemini_api_key_from_secret_manager

# Import the raw context loader from root_agent.py
# This assumes root_agent.py defines load_raw_adk_context_for_root_agent
# and it's suitable for reuse, or create a specific one here.

from .context_loader import get_escaped_adk_context_for_llm, ADK_CONTEXT_DATA_FILE

get_gemini_api_key_from_secret_manager()

# List of boilerplate strings/patterns to remove from GitHub issue content
# We'll use regex for more robust matching (e.g., ignoring leading/trailing spaces, case-insensitivity)
# Note: Regex special characters within these strings need to be escaped (e.g., `.` becomes `\.`, `*` becomes `\*`)
# For simplicity, we'll start with direct string removal and then show regex.

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
    """Removes predefined boilerplate strings from the GitHub issue text."""
    if not text:
        return ""
    cleaned_text = text
    for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(boilerplate, "")
    
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip() # Consolidate multiple newlines
    return cleaned_text

def github_issue_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm_with_note = get_escaped_adk_context_for_llm()
    
    last_event = None
    raw_browser_tool_output = "" # Store the raw output first

    if hasattr(context, '_invocation_context') and \
       hasattr(context._invocation_context, 'session') and \
       context._invocation_context.session and \
       context._invocation_context.session.events:
        last_event = context._invocation_context.session.events[-1]

        if last_event and last_event.get_function_responses():
            for func_response in last_event.get_function_responses():
                if func_response.name == browser_agent.name: 
                    if isinstance(func_response.response, dict) and "extracted_details" in func_response.response:
                        raw_browser_tool_output = func_response.response["extracted_details"]
                        break
    
    # Clean the browser tool output if it exists
    cleaned_browser_tool_output = ""
    if raw_browser_tool_output:
        cleaned_browser_tool_output = clean_github_issue_text(raw_browser_tool_output)

    if cleaned_browser_tool_output: # Check cleaned output
        instruction_phase = f"""
{cleaned_browser_tool_output}
"""
    else:
        # If raw_browser_tool_output was present but cleaned_browser_tool_output is empty,
        # it implies the content was *only* boilerplate. We might still want to inform the user.
        # Or, if no output was fetched at all, proceed to ask/call tool.
        if raw_browser_tool_output and not cleaned_browser_tool_output: # Content was only boilerplate
            instruction_phase = """
The GitHub issue content fetched by the browser agent appears to consist only of template boilerplate text. 
Please inform the user that the issue content was empty or only contained template text, and ask if they can provide more specific details directly.
This is your final answer for this query.
"""
        else: # No browser output yet, ask for issue or call tool
            instruction_phase = """
Your task is to handle a GitHub issue.
1.  Examine the user's request. If an issue number for 'google/adk-python' is mentioned (e.g., "issue 123", "GitHub ticket 456"), make a note of the number.
2.  If you do not have an issue number (it is not in the query and not in your memory from a previous turn), YOU MUST ask the user for "the GitHub issue number for 'google/adk-python'". This should be your only response for this turn. Do not call any tools yet.
3.  If you have the issue number:
    a. Construct the URL: `https://github.com/google/adk-python/issues/THE_ISSUE_NUMBER_YOU_HAVE`.
    b. Call the 'browser_utility_agent' tool with this exact URL. This will be your only action for this turn. Do not add any conversational text before or after the tool call.
"""

    base_instruction = f"""
You are a specialized agent that knows the Google's ADK version 0.5.0 in detail.
Your knowledge base includes the following ADK context:
--- START OF ADK CONTEXT ---
{{ADK_CONTEXT_PLACEHOLDER}}
--- END OF ADK CONTEXT ---

{instruction_phase}
"""
    final_instruction = base_instruction.replace(
        "{ADK_CONTEXT_PLACEHOLDER}", 
        adk_context_for_llm_with_note.replace('%', '%%')
    )
    return final_instruction

github_issue_solver_agent = ADKAgent(
    name="github_issue_solver_agent",
    model=Gemini(model="gemini-2.5-pro-preview-05-06"),
    instruction=github_issue_agent_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) 
    ],
    disallow_transfer_to_peers=True, 
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=65536,
        top_p=0.6
    )
)