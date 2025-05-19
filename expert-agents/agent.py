# expert-agents/agent.py
import os
import logging
import re
import json
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from google.genai import types as genai_types

from .context_loader import get_escaped_adk_context_for_llm
from .github_issue_fetcher_agent import github_issue_fetcher_agent, GitHubIssueFetcherToolInput
from .adk_guidance_agent import adk_guidance_agent, AdkGuidanceInput
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

TRIGGER_GUIDANCE_TOOL_NAME = "internal_trigger_adk_guidance_with_context"

async def root_agent_after_tool_callback(
    tool: BaseTool, 
    args: dict,
    tool_context: ToolContext, 
    tool_response: dict | str # For AgentTool, this is the content of FunctionResponse.response
) -> genai_types.Content | None:
    
    invocation_ctx_for_callback = tool_context._invocation_context

    if tool.name == github_issue_fetcher_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_fetcher_agent.name}'. Raw tool_response: {str(tool_response)[:500]}")
        
        fetcher_output_text = ""
        # ** THE CRUCIAL FIX IS HERE **
        if isinstance(tool_response, dict):
            # AgentTool often wraps the sub-agent's Content object in a 'result' field
            # if the sub-agent's callback returned Content.
            # Or, it might put the direct text output under a 'text' field.
            if "result" in tool_response and isinstance(tool_response["result"], genai_types.Content):
                content_obj = tool_response["result"]
                if content_obj.parts and content_obj.parts[0].text:
                    fetcher_output_text = content_obj.parts[0].text
                    logger.info(f"RootAgent (after_tool_callback): Extracted text from tool_response['result'].Content: '{fetcher_output_text[:100]}...'")
            elif "text" in tool_response and isinstance(tool_response["text"], str):
                fetcher_output_text = tool_response["text"]
                logger.info(f"RootAgent (after_tool_callback): Extracted text from tool_response['text']: '{fetcher_output_text[:100]}...'")
            else:
                logger.warning(f"RootAgent (after_tool_callback): tool_response dict from {tool.name} did not have expected 'result' (as Content) or 'text' key. Dict: {tool_response}")
        elif isinstance(tool_response, str):
            fetcher_output_text = tool_response
            logger.info(f"RootAgent (after_tool_callback): tool_response from {tool.name} was a direct string: '{fetcher_output_text[:100]}...'")
        else:
            logger.error(f"RootAgent (after_tool_callback): Unexpected response format from {github_issue_fetcher_agent.name}: {type(tool_response)}")
            invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_fetcher_error"
            invocation_ctx_for_callback.session.state["temp:fetcher_error_message"] = "Error: Could not process fetcher agent response due to unexpected format."
            return None

        if not fetcher_output_text: # If extraction failed
            logger.warning(f"RootAgent (after_tool_callback): Could not extract text from {github_issue_fetcher_agent.name} response.")
            invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_fetcher_error"
            invocation_ctx_for_callback.session.state["temp:fetcher_error_message"] = "Error: Fetcher agent returned an empty or unreadable response."
            return None


        # Now check content of fetcher_output_text
        if "---BEGIN CLEANED ISSUE TEXT---" in fetcher_output_text:
            match = re.search(r"---BEGIN CLEANED ISSUE TEXT---\n(.*?)\n---END CLEANED ISSUE TEXT---", fetcher_output_text, re.DOTALL)
            if match:
                cleaned_details = match.group(1).strip()
                logger.info(f"RootAgent (after_tool_callback): Storing cleaned details. Action: call_guidance_trigger. Details: {cleaned_details[:100]}...")
                invocation_ctx_for_callback.session.state["temp:cleaned_details_for_guidance"] = cleaned_details
                invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "call_guidance_trigger"
            else: 
                logger.error("RootAgent (after_tool_callback): Markers found but could not parse details from fetcher_output_text.")
                invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_fetcher_error"
                invocation_ctx_for_callback.session.state["temp:fetcher_error_message"] = "Error: System could not parse the fetched issue details despite markers."
        else: # This means fetcher returned "ask for number", "empty/boilerplate", or an error from its own callback
            logger.info(f"RootAgent (after_tool_callback): Fetcher returned a direct message (no markers): '{fetcher_output_text}'. Storing to relay.")
            invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_fetcher_message"
            invocation_ctx_for_callback.session.state["temp:fetcher_message"] = fetcher_output_text
        
        return None # Re-prompt root_agent's LLM based on the new state

    elif tool.name == TRIGGER_GUIDANCE_TOOL_NAME:
        logger.info(f"RootAgent (after_tool_callback): Caught '{TRIGGER_GUIDANCE_TOOL_NAME}' trigger.")
        cleaned_details = invocation_ctx_for_callback.session.state.get("temp:cleaned_details_for_guidance")
        
        invocation_ctx_for_callback.session.state.pop("temp:cleaned_details_for_guidance", None)
        invocation_ctx_for_callback.session.state.pop("temp:action_for_next_turn", None)

        if not cleaned_details:
            logger.error("RootAgent (after_tool_callback): No cleaned_details found in state for guidance trigger.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Cleaned details not found for guidance.")])

        guidance_agent_tool_instance = next((t for t in root_agent.tools if isinstance(t, AgentTool) and t.agent.name == adk_guidance_agent.name), None)
        
        if not guidance_agent_tool_instance:
            logger.error("RootAgent (after_tool_callback): Could not find adk_guidance_agent tool instance.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Guidance agent tool not found.")])

        logger.info(f"RootAgent (after_tool_callback): Manually invoking adk_guidance_agent with details: {cleaned_details[:100]}...")
        try:
            minimal_tool_context_for_guidance_call = ToolContext(invocation_ctx_for_callback)
            guidance_response_dict = await guidance_agent_tool_instance.run_async(
                args={"document_text": cleaned_details}, 
                tool_context=minimal_tool_context_for_guidance_call 
            )
            guidance_text = (guidance_response_dict or {}).get("text", "No guidance was provided or an error occurred from the guidance agent.")
            logger.info(f"RootAgent (after_tool_callback): Guidance received: {guidance_text[:100]}... Returning as final output.")
            return genai_types.Content(parts=[genai_types.Part(text=guidance_text)])
        except Exception as e:
            logger.error(f"RootAgent (after_tool_callback): Error manually invoking adk_guidance_agent: {e}", exc_info=True)
            return genai_types.Content(parts=[genai_types.Part(text=f"Internal Error: Could not get guidance due to: {e}")])
    
    logger.warning(f"RootAgent (after_tool_callback): Callback triggered for unhandled tool: {tool.name}")
    return None

# --- root_agent_instruction_provider and other definitions remain the same ---
# (No changes needed for root_agent_instruction_provider from the previous version
# as it relies on the state being correctly set by the after_tool_callback)
def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context
    
    user_query_text = ""
    session_state = {} 
    if invocation_ctx:
        if hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
            user_query_text = get_text_from_content(invocation_ctx.user_content)
        if hasattr(invocation_ctx, 'session') and invocation_ctx.session:
            session_state = dict(invocation_ctx.session.state) 

    action_for_next_turn = session_state.get("temp:action_for_next_turn")

    if action_for_next_turn == "call_guidance_trigger":
        logger.info("RootAgent (instruction_provider): State indicates cleaned details ready. Instructing LLM to call trigger_guidance.")
        system_instruction = f"""
You are an expert orchestrator. Details for a GitHub issue have been fetched and are ready for ADK-specific guidance.
Your ONLY task now is to call the function '{TRIGGER_GUIDANCE_TOOL_NAME}'. Do not provide any arguments to it.
This call will trigger the ADK guidance. Do not add any other text.
"""
        if invocation_ctx and hasattr(invocation_ctx, 'session'):
            invocation_ctx.session.state.pop("temp:action_for_next_turn", None)

    elif action_for_next_turn == "relay_fetcher_message" or \
         action_for_next_turn == "relay_fetcher_error":
        fetcher_message = session_state.get("temp:fetcher_message", "An unknown issue occurred with fetching details.")
        logger.info(f"RootAgent (instruction_provider): State indicates relaying fetcher message: {fetcher_message}")
        system_instruction = f"Your final response for this turn MUST be exactly: '{fetcher_message}'"
        if invocation_ctx and hasattr(invocation_ctx, 'session'): 
            invocation_ctx.session.state.pop("temp:action_for_next_turn", None)
            invocation_ctx.session.state.pop("temp:fetcher_message", None)
            invocation_ctx.session.state.pop("temp:cleaned_details_for_guidance", None) 
    else: 
        patterns = [
            re.compile(r"(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)(?:\s*(?:on|for|in|related to)\s*google/adk-python)?", re.IGNORECASE),
            re.compile(r"google/adk-python\s*(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)", re.IGNORECASE),
            re.compile(r"(\d+)\s*(?:on|for|in|related to)\s*google/adk-python", re.IGNORECASE)
        ]
        extracted_issue_number = None
        for pattern in patterns:
            match = pattern.search(user_query_text)
            if match:
                for group_val in match.groups():
                    if group_val and group_val.isdigit():
                        extracted_issue_number = group_val
                        break
            if extracted_issue_number:
                break
        
        is_github_keywords_present = "github" in user_query_text.lower() or \
                                     any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature"])

        if extracted_issue_number:
            logger.info(f"RootAgent (instruction_provider): Found issue number '{extracted_issue_number}'. Calling fetcher.")
            tool_input_obj = GitHubIssueFetcherToolInput(issue_number=extracted_issue_number)
            tool_arg_json_str = tool_input_obj.model_dump_json()
            
            system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking about GitHub issue number {extracted_issue_number} for 'google/adk-python'.
Your task is to call the '{github_issue_fetcher_agent.name}' tool.
You MUST pass the issue number to the tool using the following JSON argument format:
{tool_arg_json_str}
This is your only action for this turn.
"""
        elif is_github_keywords_present:
            logger.info("RootAgent (instruction_provider): GitHub keywords present, but no issue number. Asking.")
            system_instruction = "Your final response for this turn MUST be exactly: 'It looks like you're asking about a GitHub issue for google/adk-python, but I couldn't find a specific issue number. Please provide the GitHub issue number.'"
        else: 
            logger.info(f"RootAgent (instruction_provider): General ADK query: '{user_query_text}'")
            system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 0.5.0.
Your primary role is to answer general questions about ADK.
When a user starts a conversation, greet them by introducing yourself as an ADK 0.5.0 expert.
Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general ADK questions):
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
"""
    return system_instruction

API_KEY = get_gemini_api_key_from_secret_manager()

class InternalTriggerGuidanceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name=TRIGGER_GUIDANCE_TOOL_NAME,
            description="Internal trigger to proceed with ADK guidance based on fetched context."
        )
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=None 
        )
    async def run_async(self, args: dict, tool_context: ToolContext) -> dict:
        return {"status": "Guidance trigger successfully received by internal callback."}

root_agent_tools = [
    AgentTool(agent=github_issue_fetcher_agent),
    AgentTool(agent=adk_guidance_agent),
    InternalTriggerGuidanceTool()
]

root_agent = ADKAgent(
    name="adk_expert_orchestrator",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=root_agent_instruction_provider,
    tools=root_agent_tools,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=root_agent_after_tool_callback,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=60000, 
        top_p=0.6,
    )
)