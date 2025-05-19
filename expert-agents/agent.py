# expert-agents/agent.py
import os
import logging
import re
import json
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext # For root_agent's after_tool_callback
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool # For type hinting in callback
from google.adk.tools.tool_context import ToolContext # For type hinting in callback


from google.genai import types as genai_types

from .context_loader import get_escaped_adk_context_for_llm
from .github_issue_fetcher_agent import github_issue_fetcher_agent, GitHubIssueFetcherInput
from .adk_guidance_agent import adk_guidance_agent, AdkGuidanceInput
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

load_dotenv()
logger = logging.getLogger(__name__)

# --- Helper function to get text from Content object ---
def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# --- Placeholder tool name for triggering guidance ---
TRIGGER_GUIDANCE_TOOL_NAME = "internal_trigger_adk_guidance_with_context"

# --- Root Agent's after_tool_callback ---
async def root_agent_after_tool_callback(
    tool: BaseTool, # The tool that was just called by root_agent's LLM
    args: dict,
    tool_context: ToolContext, # The context of the root_agent when its LLM made the call
    tool_response: dict | str # The response from the tool that was just called
) -> genai_types.Content | None:
    
    # Case 1: github_issue_fetcher_agent just ran
    if tool.name == github_issue_fetcher_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_fetcher_agent.name}'.")
        
        fetcher_output_text = ""
        if isinstance(tool_response, dict) and "text" in tool_response:
            fetcher_output_text = tool_response["text"]
        elif isinstance(tool_response, str):
            fetcher_output_text = tool_response
        else:
            logger.error(f"RootAgent: Unexpected response format from {github_issue_fetcher_agent.name}: {tool_response}")
            return genai_types.Content(parts=[genai_types.Part(text="Error: Could not process fetcher agent response.")])

        if "Please provide the GitHub issue number" in fetcher_output_text:
            logger.info("RootAgent (after_tool_callback): Fetcher needs issue number. Relaying to user.")
            # Let root_agent's LLM handle relaying this message in the next turn based on instruction_provider
            tool_context.state["temp:fetcher_response_type"] = "ask_issue_number"
            tool_context.state["temp:fetcher_message"] = fetcher_output_text
            return None # Allow normal LLM processing for root_agent
            
        elif "The fetched GitHub issue content appears to be empty" in fetcher_output_text or \
             "Error: Could not parse details from fetcher" in fetcher_output_text or \
             "The browser tool could not fetch or extract content" in fetcher_output_text:
            logger.info("RootAgent (after_tool_callback): Fetcher reported empty/error. Relaying to user.")
            tool_context.state["temp:fetcher_response_type"] = "empty_or_error"
            tool_context.state["temp:fetcher_message"] = fetcher_output_text
            return None # Allow normal LLM processing for root_agent

        elif "---BEGIN CLEANED ISSUE TEXT---" in fetcher_output_text:
            match = re.search(r"---BEGIN CLEANED ISSUE TEXT---\n(.*?)\n---END CLEANED ISSUE TEXT---", fetcher_output_text, re.DOTALL)
            if match:
                cleaned_details = match.group(1).strip()
                logger.info(f"RootAgent (after_tool_callback): Storing cleaned details for guidance. Details: {cleaned_details[:100]}...")
                tool_context.state["temp:cleaned_details_for_guidance"] = cleaned_details
                # original_user_query should be available from the initial invocation context if needed
                # For simplicity, let's assume the root_agent's LLM will be re-prompted to call the trigger tool
                tool_context.state["temp:fetcher_response_type"] = "details_ready"
                return None # Allow normal LLM processing; instruction_provider will see this state
            else:
                logger.error("RootAgent (after_tool_callback): Could not parse details from fetcher_output_text despite markers.")
                tool_context.state["temp:fetcher_response_type"] = "parsing_error"
                tool_context.state["temp:fetcher_message"] = "Error: System could not parse the fetched issue details."
                return None
        else:
            # Unexpected output from fetcher
            logger.warning(f"RootAgent (after_tool_callback): Unexpected output from fetcher: {fetcher_output_text[:200]}")
            tool_context.state["temp:fetcher_response_type"] = "unexpected"
            tool_context.state["temp:fetcher_message"] = fetcher_output_text
            return None

    # Case 2: The root_agent's LLM just "called" our placeholder trigger tool
    elif tool.name == TRIGGER_GUIDANCE_TOOL_NAME: # This name matches the one we instruct the LLM to call
        logger.info(f"RootAgent (after_tool_callback): Caught '{TRIGGER_GUIDANCE_TOOL_NAME}' trigger.")
        cleaned_details = tool_context.state.get("temp:cleaned_details_for_guidance")
        
        if not cleaned_details:
            logger.error("RootAgent (after_tool_callback): No cleaned_details found in state for guidance trigger.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Cleaned details not found for guidance.")])

        # Manually prepare input for adk_guidance_agent
        guidance_input = AdkGuidanceInput(document_text=cleaned_details)
        guidance_input_json_str = guidance_input.model_dump_json()
        
        # Manually invoke adk_guidance_agent (as an AgentTool)
        # We need to get the actual AgentTool instance
        guidance_agent_tool_instance = None
        for t in root_agent.tools: # root_agent is accessible globally here
            if isinstance(t, AgentTool) and t.agent.name == adk_guidance_agent.name:
                guidance_agent_tool_instance = t
                break
        
        if not guidance_agent_tool_instance:
            logger.error("RootAgent (after_tool_callback): Could not find adk_guidance_agent tool instance.")
            return genai_types.Content(parts=[genai_types.Part(text="Internal Error: Guidance agent tool not found.")])

        logger.info(f"RootAgent (after_tool_callback): Manually invoking adk_guidance_agent with input: {guidance_input_json_str[:200]}...")
        
        # The ToolContext for the guidance_agent_tool_instance will be fresh.
        # Its user_content will be what we pass in `args`.
        # ADK's AgentTool expects args as a dict. The LlmAgent (adk_guidance_agent)
        # with an input_schema expects that dict to be the JSON string of its input schema.
        # However, AgentTool's run_async internally creates a Content object from args.
        # If input_schema is set on adk_guidance_agent, AgentTool will try to validate `args`
        # against that schema. If `args` is `{"document_text": "..."}`, it should work.
        try:
            # AgentTool handles creating the user_content for the sub-agent from these args
            guidance_response_dict = await guidance_agent_tool_instance.run_async(
                args={"document_text": cleaned_details}, # Pass as dict, AgentTool will handle it for input_schema
                tool_context=tool_context # Pass root_agent's context, though guidance_agent may not use much of it
            )
            
            guidance_text = ""
            if isinstance(guidance_response_dict, dict) and "text" in guidance_response_dict:
                guidance_text = guidance_response_dict["text"]
            elif isinstance(guidance_response_dict, str): # If it directly returned a string
                guidance_text = guidance_response_dict
            else:
                logger.error(f"RootAgent: Unexpected response format from manual adk_guidance_agent call: {guidance_response_dict}")
                guidance_text = "Error: Guidance agent returned an unexpected format."

            logger.info(f"RootAgent (after_tool_callback): Guidance received: {guidance_text[:100]}... Returning as final answer.")
            # Clear temporary state
            tool_context.state.pop("temp:cleaned_details_for_guidance", None)
            tool_context.state.pop("temp:fetcher_response_type", None)
            tool_context.state.pop("temp:fetcher_message", None)
            return genai_types.Content(parts=[genai_types.Part(text=guidance_text)])
        except Exception as e:
            logger.error(f"RootAgent (after_tool_callback): Error manually invoking adk_guidance_agent: {e}", exc_info=True)
            return genai_types.Content(parts=[genai_types.Part(text=f"Internal Error: Could not get guidance: {e}")])

    return None # No override for other tool calls

# --- Root Agent's instruction_provider ---
def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    # Check temporary state set by the after_tool_callback
    fetcher_response_type = context.state.get("temp:fetcher_response_type")
    fetcher_message = context.state.get("temp:fetcher_message")
    
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context
        
    user_query_text = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        user_query_text = get_text_from_content(invocation_ctx.user_content)

    # --- Orchestration Logic ---
    if fetcher_response_type == "details_ready":
        logger.info("RootAgent (instruction_provider): Cleaned details are ready. Instructing LLM to call trigger_guidance.")
        # Instruct LLM to make the placeholder tool call.
        # The actual data passing will happen in root_agent_after_tool_callback.
        system_instruction = f"""
You are an expert orchestrator. Details for a GitHub issue have been fetched and are ready for ADK-specific guidance.
Your ONLY task now is to call the function '{TRIGGER_GUIDANCE_TOOL_NAME}'. Do not provide any arguments to it.
This call will trigger the next step in the process.
Do not add any other text or conversational fluff.
"""
        # Clear state now that we're acting on it.
        # Note: state modification in instruction_provider is not standard.
        # Better to do this in callback if it directly returns Content.
        # However, since we are returning None from callback and letting LLM run,
        # this instruction provider is the next point of control.
        if hasattr(context, '_invocation_context') and hasattr(context._invocation_context, 'session'):
             context._invocation_context.session.state.pop("temp:fetcher_response_type", None)
             context._invocation_context.session.state.pop("temp:fetcher_message", None)
             # temp:cleaned_details_for_guidance is used by the after_tool_callback for the trigger

    elif fetcher_response_type == "ask_issue_number":
        logger.info("RootAgent (instruction_provider): Fetcher needs issue number. Relaying to user.")
        system_instruction = f"Your final response for this turn MUST be exactly: '{fetcher_message}'"
        if hasattr(context, '_invocation_context') and hasattr(context._invocation_context, 'session'):
             context._invocation_context.session.state.pop("temp:fetcher_response_type", None)
             context._invocation_context.session.state.pop("temp:fetcher_message", None)

    elif fetcher_response_type in ["empty_or_error", "parsing_error", "unexpected"]:
        logger.info(f"RootAgent (instruction_provider): Fetcher reported '{fetcher_response_type}'. Relaying message.")
        system_instruction = f"Your final response for this turn MUST be exactly: '{fetcher_message}'"
        if hasattr(context, '_invocation_context') and hasattr(context._invocation_context, 'session'):
             context._invocation_context.session.state.pop("temp:fetcher_response_type", None)
             context._invocation_context.session.state.pop("temp:fetcher_message", None)
             context._invocation_context.session.state.pop("temp:cleaned_details_for_guidance", None)


    elif "github" in user_query_text.lower() or \
         any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature request"]):
        logger.info("RootAgent (instruction_provider): Detected GitHub-related query. Calling github_issue_fetcher_agent.")
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user's query is: "{user_query_text}"
Your primary action is to call the '{github_issue_fetcher_agent.name}' tool.
The tool expects a JSON argument with a single key 'user_query'.
The value for 'user_query' MUST BE EXACTLY the user's query provided above.
Construct the JSON argument like this: {{"user_query": "{user_query_text}"}}.
This is your only action for this turn. The tool will respond, and you will handle its response in the next step.
Do not add any conversational fluff before calling the tool.
"""
    else:
        logger.info("RootAgent (instruction_provider): Detected general ADK query.")
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

# Define the placeholder tool for the root_agent's LLM.
# This tool doesn't need a complex declaration for the LLM, as its 'call'
# is intercepted by the after_tool_callback.
class InternalTriggerGuidanceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name=TRIGGER_GUIDANCE_TOOL_NAME,
            description="Internal trigger to proceed with ADK guidance based on fetched context."
        )
    # No run_async needed as it's intercepted.
    # _get_declaration can be minimal or None if the LLM doesn't need to reason about its args.
    def _get_declaration(self):
        # Minimal declaration. The LLM is told to call it with no args.
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=None # Or an empty schema if preferred
        )

root_agent_tools = [
    AgentTool(agent=github_issue_fetcher_agent),
    AgentTool(agent=adk_guidance_agent),
    InternalTriggerGuidanceTool() # Add the placeholder tool
]

root_agent = ADKAgent(
    name="adk_expert_orchestrator",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=root_agent_instruction_provider,
    tools=root_agent_tools,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=root_agent_after_tool_callback, # ADDED THIS
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=60000, 
        top_p=0.6,
    )
)