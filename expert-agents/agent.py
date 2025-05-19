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
    tool_response: dict | str 
) -> genai_types.Content | None:
    
    invocation_ctx_for_callback = tool_context._invocation_context

    if tool.name == github_issue_fetcher_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_fetcher_agent.name}'. Raw response: {str(tool_response)[:500]}")
        
        # github_issue_fetcher_agent's after_tool_callback now returns Content(text=json_string).
        # AgentTool(agent=github_issue_fetcher_agent) should pass this json_string as tool_response.
        fetcher_output_json_str = None
        if isinstance(tool_response, str):
            fetcher_output_json_str = tool_response
        elif isinstance(tool_response, dict) and "text" in tool_response: # In case AgentTool wraps it
            fetcher_output_json_str = tool_response["text"]
        elif isinstance(tool_response, dict) and "result" in tool_response: # Fallback for other AgentTool wrapping
             if isinstance(tool_response["result"], genai_types.Content) and \
                tool_response["result"].parts and tool_response["result"].parts[0].text:
                 fetcher_output_json_str = tool_response["result"].parts[0].text
             elif isinstance(tool_response["result"], str):
                 fetcher_output_json_str = tool_response["result"]


        if not fetcher_output_json_str:
            logger.warning(f"RootAgent (after_tool_callback): Could not extract JSON string from {github_issue_fetcher_agent.name} response. Raw: {tool_response}")
            invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_message"
            invocation_ctx_for_callback.session.state["temp:message_to_relay"] = "Error: Fetcher agent returned an empty or unreadable response."
            return None # Re-prompt root_agent's LLM

        logger.info(f"RootAgent (after_tool_callback): Received JSON string from fetcher: '{fetcher_output_json_str[:200]}...'")

        try:
            # Expect '{"cleaned_issue_details": "..."}' or '{"error":...}' or '{"message":...}'
            payload_from_fetcher = json.loads(fetcher_output_json_str) 
            
            if isinstance(payload_from_fetcher, dict):
                if "cleaned_issue_details" in payload_from_fetcher:
                    cleaned_details = payload_from_fetcher["cleaned_issue_details"]
                    logger.info(f"RootAgent (after_tool_callback): Got cleaned details. Storing. Action: call_guidance_trigger. Details: {cleaned_details[:100]}...")
                    invocation_ctx_for_callback.session.state["temp:cleaned_details_for_guidance"] = cleaned_details
                    invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "call_guidance_trigger"
                elif "message" in payload_from_fetcher: 
                    logger.info(f"RootAgent (after_tool_callback): Fetcher returned a message: '{payload_from_fetcher['message']}'. Storing to relay.")
                    invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_message"
                    invocation_ctx_for_callback.session.state["temp:message_to_relay"] = payload_from_fetcher['message']
                elif "error" in payload_from_fetcher: 
                    logger.error(f"RootAgent (after_tool_callback): Fetcher returned an error: '{payload_from_fetcher['error']}'. Storing to relay.")
                    invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_message"
                    invocation_ctx_for_callback.session.state["temp:message_to_relay"] = payload_from_fetcher['error']
                else:
                    logger.warning(f"RootAgent (after_tool_callback): Parsed JSON from fetcher has unexpected structure: {payload_from_fetcher}")
                    invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_message"
                    invocation_ctx_for_callback.session.state["temp:message_to_relay"] = "Error: Received unexpected data structure from fetcher agent."
            else: 
                logger.warning(f"RootAgent (after_tool_callback): Fetcher output was valid JSON but not a dict: {fetcher_output_json_str}")
                invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_message"
                invocation_ctx_for_callback.session.state["temp:message_to_relay"] = "Error: Fetcher agent returned non-dictionary JSON."
        except json.JSONDecodeError:
            logger.error(f"RootAgent (after_tool_callback): Fetcher output was not valid JSON: {fetcher_output_json_str}")
            invocation_ctx_for_callback.session.state["temp:action_for_next_turn"] = "relay_message"
            invocation_ctx_for_callback.session.state["temp:message_to_relay"] = f"Error: System could not parse response from the issue fetching process. Raw: {fetcher_output_json_str[:100]}..."
        
        return None # Re-prompt root_agent's LLM based on the new state

    elif tool.name == TRIGGER_GUIDANCE_TOOL_NAME:
        # This part should now work correctly as temp:cleaned_details_for_guidance will have the actual cleaned text
        logger.info(f"RootAgent (after_tool_callback): Caught '{TRIGGER_GUIDANCE_TOOL_NAME}' trigger.")
        cleaned_details = invocation_ctx_for_callback.session.state.get("temp:cleaned_details_for_guidance")
        invocation_ctx_for_callback.session.state.pop("temp:cleaned_details_for_guidance", None)
        invocation_ctx_for_callback.session.state.pop("temp:action_for_next_turn", None)
        if not cleaned_details:
            logger.error("RootAgent (after_tool_callback): No cleaned_details found for guidance trigger.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Cleaned details missing.")])
        guidance_agent_tool_instance = next((t for t in root_agent.tools if isinstance(t, AgentTool) and t.agent.name == adk_guidance_agent.name), None)
        if not guidance_agent_tool_instance:
            logger.error("RootAgent (after_tool_callback): Guidance agent tool instance not found.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Guidance tool not configured.")])
        logger.info(f"RootAgent (after_tool_callback): Manually invoking adk_guidance_agent with: {cleaned_details[:100]}...")
        try:
            minimal_tool_context_for_guidance_call = ToolContext(invocation_ctx_for_callback)
            guidance_agent_output_string = await guidance_agent_tool_instance.run_async(
                args={"document_text": cleaned_details}, 
                tool_context=minimal_tool_context_for_guidance_call)
            guidance_text = guidance_agent_output_string if isinstance(guidance_agent_output_string, str) else str(guidance_agent_output_string)
            logger.info(f"RootAgent (after_tool_callback): Guidance received: {guidance_text[:100]}... Returning as final.")
            return genai_types.Content(parts=[genai_types.Part(text=guidance_text)])
        except Exception as e:
            logger.error(f"RootAgent (after_tool_callback): Error invoking adk_guidance_agent: {e}", exc_info=True)
            return genai_types.Content(parts=[genai_types.Part(text=f"Internal Error: Could not get guidance: {e}")])
    
    logger.warning(f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}")
    return None

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
        logger.info("RootAgent (instruction_provider): State indicates 'call_guidance_trigger'. Instructing LLM to call trigger.")
        system_instruction = f"""
You are an expert orchestrator. Issue details are ready for ADK guidance.
Your ONLY task is to call the function '{TRIGGER_GUIDANCE_TOOL_NAME}'. Do not provide arguments.
This will trigger the guidance. Do not add other text.
"""
        if invocation_ctx and hasattr(invocation_ctx, 'session'):
            invocation_ctx.session.state.pop("temp:action_for_next_turn", None) # Consume the action

    elif action_for_next_turn == "relay_message":
        message_to_relay = session_state.get("temp:message_to_relay", "An issue occurred in a sub-process.")
        logger.info(f"RootAgent (instruction_provider): State indicates 'relay_message': {message_to_relay}")
        system_instruction = f"Your final response for this turn MUST be exactly: '{message_to_relay}'"
        if invocation_ctx and hasattr(invocation_ctx, 'session'): 
            invocation_ctx.session.state.pop("temp:action_for_next_turn", None)
            invocation_ctx.session.state.pop("temp:message_to_relay", None)
            invocation_ctx.session.state.pop("temp:cleaned_details_for_guidance", None) 
    
    else: # Initial query processing
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
        max_output_tokens=8192, 
        top_p=0.6,
    )
)