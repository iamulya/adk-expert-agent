# expert-agents/sequential_issue_processor.py
import logging
import json
from pydantic import BaseModel, Field

from google.adk.agents import SequentialAgent, Agent as LlmAgent 
from google.adk.models import Gemini
from google.adk.agents.readonly_context import ReadonlyContext 
from google.adk.agents.invocation_context import InvocationContext 
from google.genai import types as genai_types

from .tools import (
    ConstructGitHubUrlTool, ConstructGitHubUrlToolInput, ConstructGitHubUrlToolOutput,
    ExtractGitHubIssueDetailsTool, ExtractGitHubIssueDetailsToolInput, ExtractionResultInput,
    HandleExtractionResultTool, HandleExtractionResultToolOutput,
    CleanGitHubIssueTextTool, CleanGitHubIssueTextToolInput, CleanGitHubIssueTextToolOutput,
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

# 1. Wrapper for ConstructGitHubUrlTool
def construct_url_instruction_provider(context: ReadonlyContext) -> str:
    user_content_str = get_user_content_text_from_readonly_context(context)
    tool_name = ConstructGitHubUrlTool().name
    if user_content_str:
        try:
            # user_content_str is expected to be json.dumps({"request": "{\"issue_number\":\"123\"}"})
            outer_input_dict = json.loads(user_content_str)
            # actual_input_json_str is "{\"issue_number\":\"123\"}"
            actual_input_json_str = outer_input_dict.get("request")
            if actual_input_json_str is None: # Handle case where "request" key might be missing
                 logger.error(f"{tool_name} Wrapper: 'request' key missing in outer input dict: {outer_input_dict}")
                 raise ValueError("'request' key missing in input")

            # Now, actual_input_json_str should be a string that is valid JSON for GitHubIssueProcessingInput
            input_data = GitHubIssueProcessingInput.model_validate_json(actual_input_json_str)
            
            tool_input_obj = ConstructGitHubUrlToolInput(issue_number=input_data.issue_number)
            tool_arg_json = tool_input_obj.model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."
        except Exception as e:
            logger.error(f"{tool_name} Wrapper: Error parsing input: {e}, content received: '{user_content_str}'", exc_info=True)
            # Fallback: instruct LLM to call tool with an error indicator or minimal valid args
            # The tool itself should handle malformed/missing inputs gracefully.
            error_tool_args = ConstructGitHubUrlToolInput(issue_number="ERROR_PARSING_INITIAL_INPUT").model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with arguments {error_tool_args}. Input parsing failed. Output only the tool call."
            
    # Fallback if no user_content_str
    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
    empty_tool_args = ConstructGitHubUrlToolInput(issue_number="MISSING_INPUT").model_dump_json()
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. No input provided. Output only the tool call."

construct_url_wrapper_agent = LlmAgent(
    name="ConstructUrlWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=construct_url_instruction_provider,
    tools=[ConstructGitHubUrlTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# 2. Wrapper for ExtractGitHubIssueDetailsTool
def extract_details_instruction_provider(context: ReadonlyContext) -> str:
    user_content_json_str = get_user_content_text_from_readonly_context(context) 
    tool_name = ExtractGitHubIssueDetailsTool().name
    if user_content_json_str:
        try:
            # user_content_json_str is the JSON output of ConstructGitHubUrlTool (ConstructGitHubUrlToolOutput)
            # It will be like: {"url": "...", "error": "..."}
            # This dict is directly usable as args for ExtractGitHubIssueDetailsTool
            # as its input schema ExtractGitHubIssueDetailsToolInput matches this.
            tool_arg_json = user_content_json_str # Pass the JSON string directly
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."
        except Exception as e: # Should not happen if previous tool outputs valid JSON
            logger.error(f"{tool_name} Wrapper: Error processing input: {e}, content: {user_content_json_str}", exc_info=True)
            error_args = ExtractGitHubIssueDetailsToolInput(url="", error=f"Input processing failed for {tool_name}: {e}").model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with arguments {error_args}. Output only the tool call."
    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
    empty_tool_args = ExtractGitHubIssueDetailsToolInput(url="", error="No input from previous step").model_dump_json()
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. Output only the tool call."

extract_details_wrapper_agent = LlmAgent(
    name="ExtractDetailsWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=extract_details_instruction_provider,
    tools=[ExtractGitHubIssueDetailsTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# 3. Wrapper for HandleExtractionResultTool
def handle_extraction_instruction_provider(context: ReadonlyContext) -> str:
    user_content_json_str = get_user_content_text_from_readonly_context(context)
    tool_name = HandleExtractionResultTool().name
    if user_content_json_str:
        # user_content_json_str is JSON of ExtractionResultInput (output of ExtractGitHubIssueDetailsTool)
        # Tool expects args matching ExtractionResultInput schema.
        return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {user_content_json_str}. Output only the tool call."
    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
    empty_tool_args = ExtractionResultInput(error="No input from previous step").model_dump_json(exclude_none=True)
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. Output only the tool call."

handle_extraction_wrapper_agent = LlmAgent(
    name="HandleExtractionWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=handle_extraction_instruction_provider,
    tools=[HandleExtractionResultTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# 4. Wrapper for CleanGitHubIssueTextTool
def clean_text_instruction_provider(context: ReadonlyContext) -> str:
    user_content_json_str = get_user_content_text_from_readonly_context(context)
    tool_name = CleanGitHubIssueTextTool().name
    if user_content_json_str:
        try:
            # user_content_json_str is JSON of HandleExtractionResultToolOutput {"text_content": "..."}
            input_dict_from_prev_tool = json.loads(user_content_json_str)
            raw_text = input_dict_from_prev_tool.get("text_content", "") 
            
            tool_input_obj = CleanGitHubIssueTextToolInput(raw_text=raw_text)
            tool_arg_json = tool_input_obj.model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."
        except Exception as e:
            logger.error(f"{tool_name} Wrapper: Error parsing input: {e}, content: {user_content_json_str}", exc_info=True)
            error_args = CleanGitHubIssueTextToolInput(raw_text=f"Input parsing failed for {tool_name}: {e}").model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with arguments {error_args}. Output only the tool call."
    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
    empty_tool_args = CleanGitHubIssueTextToolInput(raw_text="No input from previous step").model_dump_json()
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. Output only the tool call."

clean_text_wrapper_agent = LlmAgent(
    name="CleanTextWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=clean_text_instruction_provider,
    tools=[CleanGitHubIssueTextTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# 5. Wrapper for ADKGuidanceTool
def adk_guidance_instruction_provider(context: ReadonlyContext) -> str:
    user_content_json_str = get_user_content_text_from_readonly_context(context)
    tool_name = ADKGuidanceTool().name
    if user_content_json_str:
        try:
            # user_content_json_str is JSON of CleanGitHubIssueTextToolOutput {"cleaned_text": "..."}
            input_dict_from_prev_tool = json.loads(user_content_json_str)
            cleaned_text = input_dict_from_prev_tool.get("cleaned_text", "")

            tool_input_obj = ADKGuidanceToolInput(document_text=cleaned_text)
            tool_arg_json = tool_input_obj.model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."
        except Exception as e:
            logger.error(f"{tool_name} Wrapper: Error parsing input: {e}, content: {user_content_json_str}", exc_info=True)
            error_args = ADKGuidanceToolInput(document_text=f"Input parsing failed for {tool_name}: {e}").model_dump_json()
            return f"Your task is to call the '{tool_name}' tool with arguments {error_args}. Output only the tool call."
    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
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
    generate_content_config=genai_types.GenerateContentConfig(temperature=0)
)

# --- The Sequential Agent ---
github_issue_processing_agent = SequentialAgent(
    name="github_issue_processing_sequential_agent",
    description=(
        "Processes a GitHub issue by fetching its details, cleaning the content, "
        "and providing ADK-specific guidance. Input should be a JSON string "
        "like '{\"request\": \"{\\\"issue_number\\\": \\\"123\\\"}\"}'. "
        "Output will be a JSON string containing the guidance or an error."
    ),
    sub_agents=[
        construct_url_wrapper_agent,
        extract_details_wrapper_agent,
        handle_extraction_wrapper_agent,
        clean_text_wrapper_agent,
        adk_guidance_wrapper_agent,
    ]
)