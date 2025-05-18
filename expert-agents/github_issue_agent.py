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

def github_issue_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm_with_note = get_escaped_adk_context_for_llm()
    
    last_event = None
    # Correctly access session through the internal _invocation_context of ReadonlyContext
    if hasattr(context, '_invocation_context') and \
       context._invocation_context.session and \
       context._invocation_context.session.events:
        last_event = context._invocation_context.session.events[-1]

    browser_tool_output_text = ""
    if last_event and last_event.get_function_responses():
        for func_response in last_event.get_function_responses():
            # Ensure browser_agent.name is accessible here. If browser_agent is not in scope,
            # you might need to hardcode the name or pass it differently.
            # Assuming browser_agent is the imported instance:
            if func_response.name == browser_agent.name: 
                if isinstance(func_response.response, dict) and "extracted_details" in func_response.response:
                    browser_tool_output_text = func_response.response["extracted_details"]
                    break
    
    if browser_tool_output_text:
        instruction_phase = f"""
You have received the following details from the GitHub issue:
<github_issue_details>
{browser_tool_output_text}
</github_issue_details>

Your task is now to use these details, along with your ADK knowledge (from the ADK context below), to formulate a comprehensive solution, provide insights, or answer questions about the issue. This is your final answer for this query.
"""
    else:
        instruction_phase = """
Your task is to handle a GitHub issue.
1.  Examine the user's request. If an issue number for 'google/adk-python' is mentioned, make a note of it.
2.  If you do not have an issue number (it is not in the query and not in your memory from a previous turn), YOU MUST ask the user for "the GitHub issue number". This should be your only response for this turn. Do not call any tools yet.
3.  If you have the issue number (either from the current query or a previous turn):
    a. Construct the URL: `https://github.com/google/adk-python/issues/THE_ISSUE_NUMBER_YOU_HAVE`.
    b. Call the 'browser_utility_agent' tool with this exact URL. This will be your only action for this turn.
"""

    base_instruction = f"""
You are a specialized agent for solving GitHub issues for 'google/adk-python', using Google's ADK version 0.5.0.
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
)