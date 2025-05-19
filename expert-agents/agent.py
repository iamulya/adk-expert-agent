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
from .github_issue_fetcher_agent import github_issue_fetcher_agent, GitHubIssueFetcherToolInput # Updated Input Schema
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
    
    if tool.name == github_issue_fetcher_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_fetcher_agent.name}'.")
        
        fetcher_output_text = ""
        if isinstance(tool_response, dict) and "text" in tool_response:
            fetcher_output_text = tool_response["text"]
        elif isinstance(tool_response, str):
            fetcher_output_text = tool_response
        else: # Should not happen if fetcher's callback works
            logger.error(f"RootAgent: Unexpected response format from {github_issue_fetcher_agent.name}: {tool_response}")
            return genai_types.Content(parts=[genai_types.Part(text="Error: Could not process fetcher agent response.")])

        if "---BEGIN CLEANED ISSUE TEXT---" in fetcher_output_text:
            match = re.search(r"---BEGIN CLEANED ISSUE TEXT---\n(.*?)\n---END CLEANED ISSUE TEXT---", fetcher_output_text, re.DOTALL)
            if match:
                cleaned_details = match.group(1).strip()
                logger.info(f"RootAgent (after_tool_callback): Storing cleaned details for guidance trigger. Details: {cleaned_details[:100]}...")
                tool_context.state["temp:cleaned_details_for_guidance"] = cleaned_details
                # Returning None will re-invoke root_agent's instruction_provider, which will see the state.
                return None 
            else: # Should not happen if fetcher's callback works
                logger.error("RootAgent (after_tool_callback): Could not parse details from fetcher_output_text despite markers.")
                return genai_types.Content(parts=[genai_types.Part(text="Error: System could not parse the fetched issue details.")])
        else: # This means fetcher returned an error or "empty" message
            logger.info(f"RootAgent (after_tool_callback): Fetcher returned a direct message: {fetcher_output_text}. Relaying.")
            return genai_types.Content(parts=[genai_types.Part(text=fetcher_output_text)])

    elif tool.name == TRIGGER_GUIDANCE_TOOL_NAME:
        logger.info(f"RootAgent (after_tool_callback): Caught '{TRIGGER_GUIDANCE_TOOL_NAME}' trigger.")
        cleaned_details = tool_context.state.get("temp:cleaned_details_for_guidance")
        
        if not cleaned_details:
            logger.error("RootAgent (after_tool_callback): No cleaned_details found in state for guidance trigger.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Cleaned details not found for guidance.")])

        guidance_agent_tool_instance = next((t for t in root_agent.tools if isinstance(t, AgentTool) and t.agent.name == adk_guidance_agent.name), None)
        
        if not guidance_agent_tool_instance:
            logger.error("RootAgent (after_tool_callback): Could not find adk_guidance_agent tool instance.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Guidance agent tool not found.")])

        logger.info(f"RootAgent (after_tool_callback): Manually invoking adk_guidance_agent with details: {cleaned_details[:100]}...")
        try:
            guidance_response_dict = await guidance_agent_tool_instance.run_async(
                args={"document_text": cleaned_details}, 
                tool_context=tool_context
            )
            guidance_text = (guidance_response_dict or {}).get("text", "No guidance was provided or an error occurred.")
            logger.info(f"RootAgent (after_tool_callback): Guidance received: {guidance_text[:100]}... Returning as final.")
            tool_context.state.pop("temp:cleaned_details_for_guidance", None)
            return genai_types.Content(parts=[genai_types.Part(text=guidance_text)])
        except Exception as e:
            logger.error(f"RootAgent (after_tool_callback): Error manually invoking adk_guidance_agent: {e}", exc_info=True)
            return genai_types.Content(parts=[genai_types.Part(text=f"Internal Error: Could not get guidance: {e}")])
    return None

def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context
        
    user_query_text = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        user_query_text = get_text_from_content(invocation_ctx.user_content)

    # --- State Check ---
    if "temp:cleaned_details_for_guidance" in context.state:
        logger.info("RootAgent (instruction_provider): Cleaned details ready. Instructing LLM to call trigger_guidance.")
        system_instruction = f"""
You are an expert orchestrator. Details for a GitHub issue have been fetched and are ready for ADK-specific guidance.
Your ONLY task now is to call the function '{TRIGGER_GUIDANCE_TOOL_NAME}'. Do not provide any arguments to it.
This call will trigger the ADK guidance. Do not add any other text.
"""
        # Attempt to clean up state here. Consider moving to after_tool_callback if this causes issues.
        if invocation_ctx and hasattr(invocation_ctx, 'session'):
             invocation_ctx.session.state.pop("temp:fetcher_response_type", None) # Should have been consumed by after_tool_callback
             invocation_ctx.session.state.pop("temp:fetcher_message", None) # Should have been consumed
        return system_instruction

    # --- Initial User Query Processing ---
    
    # More robust regex patterns to find an issue number
    # Pattern 1: "issue/ticket/etc. [number] ... [repo_name]" OR "[repo_name] ... issue/ticket/etc. [number]"
    # Pattern 2: "issue/ticket/etc. [number]" (more general, if repo name isn't explicitly mentioned with it)
    patterns = [
        re.compile(r"(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)(?:\s*(?:on|for|in|related to)\s*google/adk-python)?", re.IGNORECASE),
        re.compile(r"google/adk-python\s*(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(?:on|for|in|related to)\s*google/adk-python", re.IGNORECASE) # e.g. "fix for 773 on google/adk-python"
    ]
    
    extracted_issue_number = None
    for pattern in patterns:
        match = pattern.search(user_query_text)
        if match:
            # Iterate through groups to find the one that captured digits
            for group_val in match.groups():
                if group_val and group_val.isdigit():
                    extracted_issue_number = group_val
                    break
        if extracted_issue_number:
            break
            
    # Keywords that suggest a GitHub-related query, even if number extraction fails initially
    is_github_keywords_present = "github" in user_query_text.lower() or \
                                 any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature request", "report"])


    if extracted_issue_number:
        logger.info(f"RootAgent (instruction_provider): Found issue number '{extracted_issue_number}' in query '{user_query_text}'. Calling fetcher.")
        tool_arg_json = json.dumps({"issue_number": extracted_issue_number})
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking about GitHub issue number {extracted_issue_number} for 'google/adk-python'.
Your task is to call the '{github_issue_fetcher_agent.name}' tool.
You MUST pass the issue number to the tool using the following JSON argument format:
{tool_arg_json}
This is your only action for this turn. The tool will respond, and you will handle its response in the next step.
Do not add any conversational fluff.
"""
    elif is_github_keywords_present: # Keywords present, but no number extracted
        logger.info(f"RootAgent (instruction_provider): GitHub keywords present in '{user_query_text}', but no issue number. Asking user.")
        system_instruction = "Your final response for this turn MUST be exactly: 'It looks like you're asking about a GitHub issue for google/adk-python, but I couldn't find a specific issue number. Please provide the GitHub issue number.'"
    
    else: # General ADK question
        logger.info(f"RootAgent (instruction_provider): Detected general ADK query: '{user_query_text}'")
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
        # This tool's "execution" is fully handled by the root_agent's after_tool_callback.
        # It just needs to exist for the LLM to "call" it.
        # Returning an empty dict or a success message is fine.
        return {"status": "Guidance trigger received by callback."}


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