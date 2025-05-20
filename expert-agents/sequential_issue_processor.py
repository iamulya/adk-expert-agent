# expert-agents/sequential_issue_processor.py
import logging
import json
from pydantic import BaseModel, Field, ValidationError

from google.adk.agents import SequentialAgent, Agent as LlmAgent
from google.adk.models import Gemini
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool 
from google.adk.tools.tool_context import ToolContext 
from google.genai import types as genai_types

from .tools import (
    GetGithubIssueDescriptionTool,
    ADKGuidanceTool, ADKGuidanceToolInput, ADKGuidanceToolOutput,
)
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)

class GitHubIssueProcessingInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

class SequentialProcessorFinalOutput(BaseModel):
    guidance: str

def get_user_content_text_from_readonly_context(context: ReadonlyContext) -> str:
    invocation_ctx = getattr(context, '_invocation_context', None)
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            return invocation_ctx.user_content.parts[0].text
    return ""

# --- Wrapper Agents for each tool in the sequence ---

# 1. Wrapper for GetGithubIssueDescriptionTool
def get_issue_description_instruction_provider(context: ReadonlyContext) -> str:
    user_content_str = get_user_content_text_from_readonly_context(context)
    tool_name = GetGithubIssueDescriptionTool().name
    
    # user_content_str is expected to be a JSON string like "{\"issue_number\":\"123\"}" (S_inner)
    # However, due to LLM behavior or AgentTool wrapping, it might sometimes be
    # "{\"request\": \"{\\\"issue_number\\\":\\\"123\\\"}\"}" (S_outer)
    
    payload_to_validate = user_content_str

    if user_content_str:
        try:
            # Attempt to parse user_content_str as if it's the S_outer structure
            data = json.loads(user_content_str)
            if isinstance(data, dict) and "request" in data and isinstance(data["request"], str):
                # It was S_outer, extract S_inner
                payload_to_validate = data["request"]
                logger.info(f"{tool_name} Wrapper: Unwrapped 'request' from outer JSON. Using: {payload_to_validate}")
        except json.JSONDecodeError:
            # user_content_str is not a JSON string representing a dict (e.g., it's already S_inner).
            logger.warning(f"{tool_name} Wrapper: user_content_str is not a JSON dict, assuming direct payload: {user_content_str[:100]}")
            pass # payload_to_validate remains user_content_str, hopefully S_inner

        try:
            # payload_to_validate should now be S_inner ("{\"issue_number\":\"773\"}")
            input_data = GitHubIssueProcessingInput.model_validate_json(payload_to_validate)
            
            # GetGithubIssueDescriptionTool expects issue_number as an integer.
            tool_arg_dict = {"issue_number": int(input_data.issue_number)}
            # owner and repo use defaults specified in the tool's declaration
            tool_arg_json = json.dumps(tool_arg_dict)
            
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."

        except (json.JSONDecodeError, ValidationError, ValueError) as e: 
            logger.error(f"{tool_name} Wrapper: Error validating/processing payload '{payload_to_validate}': {e}", exc_info=True)
            error_tool_args = json.dumps({"issue_number": "ERROR_PROCESSING_PAYLOAD"}) # Ensure issue_number is int if tool expects
            return f"Your task is to call the '{tool_name}' tool with arguments {error_tool_args}. Payload processing failed. Output only the tool call."
            
    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
    empty_tool_args = json.dumps({"issue_number": 0}) # Default to a valid int if tool expects
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. No input provided. Output only the tool call."

get_issue_description_wrapper_agent = LlmAgent(
    name="GetIssueDescriptionWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=get_issue_description_instruction_provider,
    tools=[GetGithubIssueDescriptionTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# Callback for adk_guidance_wrapper_agent to ensure its output is a JSON string
async def adk_guidance_wrapper_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict 
) -> genai_types.Content | None:
    if tool.name == ADKGuidanceTool().name:
        json_output_str = json.dumps(tool_response)
        logger.info(f"ADKGuidanceWrapperAgent (after_tool_callback): Final output as JSON string: {json_output_str[:200]}")
        return genai_types.Content(parts=[genai_types.Part(text=json_output_str)])
    return None

# 2. Wrapper for ADKGuidanceTool
def adk_guidance_instruction_provider(context: ReadonlyContext) -> str:
    user_content_json_str = get_user_content_text_from_readonly_context(context)
    tool_name = ADKGuidanceTool().name
    
    document_text = "Error: Could not retrieve valid content from the previous step." 

    if user_content_json_str:
        try:
            input_dict_from_prev_tool = json.loads(user_content_json_str)
            
            error_message = input_dict_from_prev_tool.get("error")
            description_text = input_dict_from_prev_tool.get("description")

            if error_message is not None:
                document_text = f"Error retrieving GitHub issue: {error_message}"
            elif description_text is not None:
                document_text = description_text if description_text else "The GitHub issue description is empty."
            else:
                document_text = "Error: No description or error message found from GitHub issue retrieval."
            
            tool_input_obj = ADKGuidanceToolInput(document_text=document_text)
            tool_arg_json = tool_input_obj.model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."
        except Exception as e:
            logger.error(f"{tool_name} Wrapper: Error parsing input from previous step: {e}, content: {user_content_json_str}", exc_info=True)
            error_args = ADKGuidanceToolInput(document_text=f"Input parsing failed for {tool_name}: {e}").model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with arguments {error_args}. Output only the tool call."
            
    logger.warning(f"{tool_name} Wrapper: No user_content provided from previous step.")
    empty_tool_args = ADKGuidanceToolInput(document_text="No input from previous step").model_dump_json()
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. Output only the tool call."

adk_guidance_wrapper_agent = LlmAgent(
    name="ADKGuidanceWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME), 
    instruction=adk_guidance_instruction_provider,
    tools=[ADKGuidanceTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=adk_guidance_wrapper_after_tool_callback, 
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# --- The Sequential Agent ---
github_issue_processing_agent = SequentialAgent(
    name="github_issue_processing_sequential_agent",
    description=(
        "Processes a GitHub issue by fetching its description and providing ADK-specific guidance. "
        "Input should be a JSON string representing the issue details, like '{\"issue_number\": \"123\"}'. "
        "Output will be a JSON string containing the guidance or an error."
    ),
    sub_agents=[
        get_issue_description_wrapper_agent, 
        adk_guidance_wrapper_agent,          
    ]
)