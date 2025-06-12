"""
This module contains callback functions used by ADK agents.

Callbacks allow for custom logic to be executed at specific points in an agent's
lifecycle, such as before a model call. This is useful for logging, debugging,
or modifying requests on the fly.
"""

import logging
import json  # For pretty printing dicts if needed

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

# It's good practice to get a logger specific to this module
logger = logging.getLogger(__name__)

# Ensure basic logging is configured if ADK doesn't do it.
# This is a common pattern, but be mindful if ADK or the running environment (like adk web) already configures root logger.
# If not already configured, this will set it up.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def log_prompt_before_model_call(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    """
    A 'before_model_callback' that logs the detailed prompt being sent to the LLM.

    This function inspects the LlmRequest object to extract and log the
    system instruction, user content, function calls, and other parts of the
    prompt. It provides crucial visibility into the agent's "thought process"
    for debugging purposes.

    Args:
        callback_context: The context of the callback, containing agent information.
        llm_request: The request object being sent to the LLM.
    """
    try:
        prompt_details = []
        agent_name_to_log = "UnknownAgent"
        # Determine the name of the agent making the call for clear log messages.
        if hasattr(callback_context, "agent_name"):
            agent_name_to_log = callback_context.agent_name
        elif hasattr(callback_context, "_invocation_context") and hasattr(
            callback_context._invocation_context, "agent"
        ):
            agent_name_to_log = callback_context._invocation_context.agent.name

        # Log the system instruction, if present.
        if llm_request.config and llm_request.config.system_instruction:
            si_text = ""
            # system_instruction can be str or types.Content, so handle both.
            if isinstance(llm_request.config.system_instruction, str):
                si_text = llm_request.config.system_instruction
            elif (
                hasattr(llm_request.config.system_instruction, "parts")
                and llm_request.config.system_instruction.parts
            ):
                si_text = " ".join(
                    p.text
                    for p in llm_request.config.system_instruction.parts
                    if hasattr(p, "text") and p.text
                )
            else:
                si_text = str(llm_request.config.system_instruction)
            prompt_details.append("System Instruction:\n" + si_text)

        # Log the main contents of the prompt (the conversation history).
        if llm_request.contents:
            contents_log_parts = []
            for content_idx, content in enumerate(llm_request.contents):
                current_content_log = [
                    f"  Content {content_idx + 1} (Role: {content.role}):"
                ]
                if content.parts:
                    for part_idx, part in enumerate(content.parts):
                        part_detail = f"    Part {part_idx + 1}: "
                        # Format each part based on its type (text, function call, etc.)
                        if hasattr(part, "text") and part.text:
                            part_detail += f"Text: '{part.text}'"
                        elif hasattr(part, "function_call") and part.function_call:
                            args_str = "{}"
                            if part.function_call.args:
                                try:
                                    # Pretty-print the JSON arguments for readability.
                                    args_str = json.dumps(part.function_call.args)
                                except TypeError:  # Handle non-serializable args
                                    args_str = str(part.function_call.args)
                            part_detail += (
                                f"FunctionCall: {part.function_call.name}({args_str})"
                            )
                        elif (
                            hasattr(part, "function_response")
                            and part.function_response
                        ):
                            response_str = "{}"
                            if part.function_response.response:
                                try:
                                    response_str = json.dumps(
                                        part.function_response.response
                                    )
                                except TypeError:  # Handle non-serializable response
                                    response_str = str(part.function_response.response)
                            part_detail += f"FunctionResponse: {part.function_response.name} -> {response_str}"
                        elif hasattr(part, "inline_data") and part.inline_data:
                            part_detail += f"InlineData: MIME Type: {part.inline_data.mime_type}, Size: {len(part.inline_data.data) if part.inline_data.data else 0} bytes"
                        elif hasattr(part, "file_data") and part.file_data:
                            part_detail += f"FileData: MIME Type: {part.file_data.mime_type}, URI: {part.file_data.file_uri}"
                        else:
                            part_detail += "Unknown or empty part"
                        current_content_log.append(part_detail)
                else:
                    current_content_log.append("    No parts")
                contents_log_parts.append("\n".join(current_content_log))
            prompt_details.append("Contents:\n" + "\n".join(contents_log_parts))

        full_prompt_log = "\n---\n".join(prompt_details)

        model_name_to_log = llm_request.model or "UnknownModel"

        logger.info(
            f"Agent '{agent_name_to_log}' - Prompt to LLM ({model_name_to_log}):\n{full_prompt_log}"
        )
    except Exception as e:
        agent_name_to_log_error = "UnknownAgent"  # Default for error logging
        if hasattr(callback_context, "agent_name"):
            agent_name_to_log_error = callback_context.agent_name
        elif hasattr(callback_context, "_invocation_context") and hasattr(
            callback_context._invocation_context, "agent"
        ):
            agent_name_to_log_error = callback_context._invocation_context.agent.name
        logger.error(
            f"Error during prompt logging for agent '{agent_name_to_log_error}': {e}",
            exc_info=True,
        )
    # This callback should not return anything to avoid overriding the LLM call.
