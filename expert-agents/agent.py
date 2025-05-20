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
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager
from .sequential_issue_processor import github_issue_processing_agent, GitHubIssueProcessingInput, SequentialProcessorFinalOutput 

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

async def root_agent_after_tool_callback(
    tool: BaseTool, 
    args: dict,
    tool_context: ToolContext, 
    tool_response: str 
) -> genai_types.Content | None:
    
    invocation_ctx_for_callback = tool_context._invocation_context

    if tool.name == github_issue_processing_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_processing_agent.name}'. Type: {type(tool_response)}, Value: {str(tool_response)[:500]}")
        
        response_text = "Error: Could not process response from sequential agent."
        if isinstance(tool_response, str):
            try:
                response_dict = json.loads(tool_response)
                # Assuming the final output structure is ADKGuidanceToolOutput
                # which has a "guidance" field.
                validated_output = SequentialProcessorFinalOutput.model_validate(response_dict)
                response_text = validated_output.guidance
            except json.JSONDecodeError:
                logger.error(f"RootAgent: Failed to decode JSON response from sequential agent: {tool_response}", exc_info=True)
                if "error" in tool_response.lower() or "message" in tool_response.lower() or "empty" in tool_response.lower():
                     response_text = tool_response 
                else:
                    response_text = f"Error: Sequential agent returned non-JSON string: {tool_response[:200]}"
            except Exception as e: 
                logger.error(f"RootAgent: Error validating/extracting from sequential agent response string: {e}. Response: {tool_response}", exc_info=True)
                response_text = f"Error: Sequential agent returned an unexpected structure: {tool_response[:200]}"
        elif isinstance(tool_response, dict): 
             try:
                validated_output = SequentialProcessorFinalOutput.model_validate(tool_response)
                response_text = validated_output.guidance
             except Exception as e:
                logger.error(f"RootAgent: Error validating/extracting from sequential agent response dict: {e}. Response: {tool_response}", exc_info=True)
                response_text = f"Error: Sequential agent returned an unexpected dictionary structure: {str(tool_response)[:200]}"
        else:
            logger.error(f"RootAgent: Unexpected response type from sequential agent: {type(tool_response)}")
            response_text = f"Error: Unexpected response type from {github_issue_processing_agent.name}: {type(tool_response)}"
        
        logger.info(f"RootAgent (after_tool_callback): Relaying response from sequential agent: {response_text[:200]}...")
        return genai_types.Content(parts=[genai_types.Part(text=response_text)])
    
    logger.warning(f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}")
    return None

def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    invocation_ctx = context._invocation_context if hasattr(context, '_invocation_context') else None
    
    user_query_text = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
    
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
        logger.info(f"RootAgent (instruction_provider): Found issue number '{extracted_issue_number}'. Instructing to call GitHubIssueProcessingSequentialAgent.")
        
        tool_actual_input_obj = GitHubIssueProcessingInput(issue_number=extracted_issue_number)
        # This is the JSON string that needs to be the *value* of the "request" key
        tool_actual_input_json_str_value = tool_actual_input_obj.model_dump_json() 
        
        # The LLM needs to generate args like: {"request": "{\"issue_number\":\"123\"}"}
        # So, we construct the example string for the LLM carefully.
        # The value of "request" must be a string containing escaped JSON.
        
        # Method 1: Manually construct the example string for the LLM
        # final_tool_arg_json_str_for_llm = f'{{"request": "{tool_actual_input_json_str_value.replace("\"", "\\\"")}"}}'

        # Method 2: Use json.dumps which should handle escaping correctly if the structure is right.
        # The structure for the LLM's args should be a dict.
        # The value for the "request" key in that dict should be the string `tool_actual_input_json_str_value`.
        llm_tool_args_dict = {"request": tool_actual_input_json_str_value}
        final_tool_arg_json_str_for_llm = json.dumps(llm_tool_args_dict)
        
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking about GitHub issue number {extracted_issue_number} for 'google/adk-python'.
Your task is to call the '{github_issue_processing_agent.name}' tool.
You MUST pass the arguments to the tool using the following JSON argument format:
{final_tool_arg_json_str_for_llm}
Ensure the value for the "request" key is a JSON string.
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

root_agent_tools = [
    AgentTool(agent=github_issue_processing_agent),
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