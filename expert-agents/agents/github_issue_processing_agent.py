"""
This module defines the GitHub Issue Processing Sequential Agent.

This agent follows a strict, sequential workflow to handle GitHub issue queries:
1.  It receives an issue number from the root agent.
2.  The `GetIssueDescriptionWrapperAgent` calls a tool to fetch the issue's body
    from the GitHub API.
3.  The `ADKGuidanceWrapperAgent` takes the fetched description, combines it with
    the ADK knowledge base, and calls a tool to generate ADK-specific guidance.
4.  The `FormatOutputAgent` ensures the final guidance is correctly formatted
    as a JSON string, which is then returned to the root agent.

Using a SequentialAgent enforces this exact order of operations. Each step is
handled by a simple LLM-based "wrapper agent" whose only job is to call a
specific tool with the output of the previous step.
"""

import logging
import json
from pydantic import BaseModel, Field, ValidationError
from typing import AsyncGenerator

from google.adk.agents import SequentialAgent, Agent as LlmAgent, BaseAgent
from google.adk.models import Gemini
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types

from ..tools import (
    GetGithubIssueDescriptionTool,
    ADKGuidanceTool,
    ADKGuidanceToolInput,
)
from ..config import DEFAULT_MODEL_NAME
from ..callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)


class GitHubIssueProcessingInput(BaseModel):
    """Defines the input schema for the entire sequential agent."""

    issue_number: str = Field(
        description="The GitHub issue number for 'google/adk-python'."
    )


class SequentialProcessorFinalOutput(BaseModel):
    """Defines the final output schema for the entire sequential agent."""

    guidance: str


def get_user_content_text_from_readonly_context(context: ReadonlyContext) -> str:
    """
    Helper to safely extract the initial text input to the sequential agent from
    the invocation context.
    """
    invocation_ctx = getattr(context, "_invocation_context", None)
    if (
        invocation_ctx
        and hasattr(invocation_ctx, "user_content")
        and invocation_ctx.user_content
    ):
        if (
            invocation_ctx.user_content.parts
            and invocation_ctx.user_content.parts[0].text
        ):
            return invocation_ctx.user_content.parts[0].text
    # Fallback for complex inputs, checking the last event in the session.
    # This is useful in sequential agents where the "input" might be the output of a previous agent.
    if (
        invocation_ctx
        and hasattr(invocation_ctx, "session")
        and invocation_ctx.session.events
    ):
        last_event = invocation_ctx.session.events[-1]
        if (
            last_event.content
            and last_event.content.parts
            and last_event.content.parts[0].text
        ):
            return last_event.content.parts[0].text
        # If the last event was a function response, try to serialize its output.
        elif last_event.get_function_responses():
            try:
                return json.dumps(last_event.get_function_responses()[0].response)
            except Exception:
                pass  # Fall through if not serializable
    return ""


# --- Step 1: Wrapper for GetGithubIssueDescriptionTool ---


def get_issue_description_instruction_provider(context: ReadonlyContext) -> str:
    """
    Instruction provider for the first agent in the sequence, which fetches
    the GitHub issue description.
    """
    user_content_str = get_user_content_text_from_readonly_context(context)
    tool_name = GetGithubIssueDescriptionTool().name

    # The input from the root agent can be nested, e.g., '{"request": "{\"issue_number\": \"123\"}"}'.
    # This logic unwraps it to get to the core issue number payload.
    payload_to_validate = user_content_str
    if user_content_str:
        try:
            data = json.loads(user_content_str)
            if (
                isinstance(data, dict)
                and "request" in data
                and isinstance(data["request"], str)
            ):
                payload_to_validate = data["request"]
                logger.info(
                    f"{tool_name} Wrapper: Unwrapped 'request' from outer JSON. Using: {payload_to_validate}"
                )
        except json.JSONDecodeError:
            logger.warning(
                f"{tool_name} Wrapper: user_content_str is not a JSON dict, assuming direct payload: {user_content_str[:100]}"
            )
            pass

        try:
            # Validate the payload and construct the arguments for the tool call.
            input_data = GitHubIssueProcessingInput.model_validate_json(
                payload_to_validate
            )
            tool_arg_dict = {"issue_number": int(input_data.issue_number)}
            tool_arg_json = json.dumps(tool_arg_dict)
            # This specific instruction forces the LLM to call the tool.
            return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(
                f"{tool_name} Wrapper: Error validating/processing payload '{payload_to_validate}': {e}",
                exc_info=True,
            )
            error_tool_args = json.dumps({"issue_number": 0})
            return f"Your task is to call the '{tool_name}' tool with arguments {error_tool_args}. Payload processing failed. Output only the tool call."

    logger.warning(f"{tool_name} Wrapper: No user_content provided.")
    empty_tool_args = json.dumps({"issue_number": 0})
    return f"Your task is to call the '{tool_name}' tool with arguments {empty_tool_args}. No input provided. Output only the tool call."


get_issue_description_wrapper_agent = LlmAgent(
    name="GetIssueDescriptionWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=get_issue_description_instruction_provider,
    tools=[GetGithubIssueDescriptionTool()],
    # These flags prevent the agent from breaking out of the sequence.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0),
)


# --- Step 2: Wrapper for ADKGuidanceTool ---
def adk_guidance_instruction_provider(context: ReadonlyContext) -> str:
    """
    Instruction provider for the second agent, which generates ADK guidance.
    Its input is the output from the previous agent in the sequence.
    """
    # The relevant input for this agent is the *last event* from the previous agent.
    invocation_ctx = getattr(context, "_invocation_context", None)
    tool_response_json_str = ""
    if (
        invocation_ctx
        and hasattr(invocation_ctx, "session")
        and invocation_ctx.session.events
    ):
        last_event = invocation_ctx.session.events[-1]
        # We expect the last event to be a FunctionResponse from GetIssueDescriptionWrapperAgent.
        if (
            last_event.get_function_responses()
            and last_event.get_function_responses()[0].name
            == GetGithubIssueDescriptionTool().name
        ):
            try:
                # Serialize the previous tool's response to pass to the next tool.
                tool_response_json_str = json.dumps(
                    last_event.get_function_responses()[0].response
                )
            except Exception as e:
                logger.error(
                    f"ADKGuidanceWrapper: Error stringifying previous tool response: {e}",
                    exc_info=True,
                )
                tool_response_json_str = json.dumps(
                    {"error": f"Could not process previous tool response: {e}"}
                )
        else:
            logger.warning(
                f"ADKGuidanceWrapper: Last event was not the expected function response. Last event: {str(last_event)[:200]}"
            )
            tool_response_json_str = json.dumps(
                {"error": "Previous step did not return expected GitHub issue data."}
            )
    else:
        tool_response_json_str = json.dumps(
            {"error": "No previous event data found for ADKGuidanceTool."}
        )

    tool_name = ADKGuidanceTool().name
    document_text = "Error: Could not retrieve valid content from the previous step."
    tool_arg_json = ADKGuidanceToolInput(document_text=document_text).model_dump_json()

    if tool_response_json_str:
        try:
            # Parse the previous tool's output to get the issue description.
            input_dict_from_prev_tool = json.loads(tool_response_json_str)

            error_message = input_dict_from_prev_tool.get("error")
            description_text = input_dict_from_prev_tool.get("description")

            if error_message is not None:
                document_text = f"Error retrieving GitHub issue: {error_message}"
            elif description_text is not None:
                document_text = (
                    description_text
                    if description_text
                    else "The GitHub issue description is empty."
                )
            else:
                logger.warning(
                    f"{tool_name} Wrapper: Previous tool output dict '{input_dict_from_prev_tool}' did not contain 'description' or 'error' keys."
                )
                document_text = "Error: No description or error message found from GitHub issue retrieval step."

            # Prepare the arguments for the ADKGuidanceTool.
            tool_input_obj = ADKGuidanceToolInput(document_text=document_text)
            tool_arg_json = tool_input_obj.model_dump_json()
        except Exception as e:  # Broad exception for safety
            logger.error(
                f"{tool_name} Wrapper: Error processing tool_response_json_str '{tool_response_json_str}': {e}",
                exc_info=True,
            )
            # tool_arg_json will use the default error document_text

    logger.info(
        f"{tool_name} Wrapper: Using document_text: '{document_text[:100]}...' for {tool_name}"
    )
    # Instruct the LLM to call the guidance tool.
    return f"Your task is to call the '{tool_name}' tool with the following JSON arguments: {tool_arg_json}. Output only the tool call."


adk_guidance_wrapper_agent = LlmAgent(
    name="ADKGuidanceWrapperAgent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=adk_guidance_instruction_provider,
    tools=[ADKGuidanceTool()],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0),
)


# --- Step 3: FormatOutputAgent to prepare the final response ---
class FormatOutputAgent(BaseAgent):
    """
    A non-LLM agent that takes the output from the final tool call,
    validates it, and formats it into the definitive JSON string output
    for the entire sequential agent. This is a critical step to ensure the
    final output has a consistent and expected structure.
    """

    name: str = "FormatFinalOutputAgent"
    description: str = "Formats the processed data into the final JSON string output."

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """The execution logic for this non-LLM agent."""
        logger.info(
            f"{self.name}: Current session events count: {len(ctx.session.events)}"
        )
        final_output_dict = {
            "guidance": "Error: Could not extract guidance from previous step (FormatOutputAgent)."
        }

        # The input for this agent is the last event from the PREVIOUS agent in the sequence.
        if ctx.session.events:
            last_event_from_previous_agent = None
            # Iterate backwards to find the last event from ADKGuidanceWrapperAgent that contains the function response.
            for event in reversed(ctx.session.events):
                if (
                    event.author == adk_guidance_wrapper_agent.name
                    and event.get_function_responses()
                ):
                    last_event_from_previous_agent = event
                    break

            if last_event_from_previous_agent:
                logger.info(
                    f"{self.name}: Found last event from previous agent: {str(last_event_from_previous_agent.content)[:200]}"
                )
                for part in last_event_from_previous_agent.content.parts:
                    if (
                        part.function_response
                        and part.function_response.name == ADKGuidanceTool().name
                    ):
                        # Extract the dictionary response from the ADKGuidanceTool.
                        if isinstance(part.function_response.response, dict):
                            final_output_dict = part.function_response.response
                            logger.info(
                                f"{self.name}: Extracted dict from function response: {str(final_output_dict)[:200]}"
                            )
                        else:
                            logger.warning(
                                f"{self.name}: function_response.response was not a dict: {part.function_response.response}"
                            )
                        break
                else:  # Inner loop didn't break
                    logger.warning(
                        f"{self.name}: No matching function response from '{ADKGuidanceTool().name}' found in event from '{adk_guidance_wrapper_agent.name}'."
                    )
            else:
                logger.warning(
                    f"{self.name}: Could not find a relevant event from '{adk_guidance_wrapper_agent.name}'."
                )
        else:
            logger.warning(
                f"{self.name}: No events in session to process for final formatting."
            )

        try:
            # Validate the final data against the output schema and serialize to JSON.
            validated_data_for_json = SequentialProcessorFinalOutput.model_validate(
                final_output_dict
            )
            final_json_string = validated_data_for_json.model_dump_json()
        except Exception as e:
            logger.error(
                f"{self.name}: Error validating or dumping final_output_dict {str(final_output_dict)[:200]}: {e}",
                exc_info=True,
            )
            error_guidance = final_output_dict.get(
                "guidance", f"Error formatting final output: {e}"
            )
            final_json_string = json.dumps({"guidance": str(error_guidance)})

        logger.info(
            f"{self.name}: Yielding final JSON string: {final_json_string[:200]}"
        )
        # Yield the final, formatted event. This becomes the terminal output of the SequentialAgent.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=final_json_string)]
            ),
        )


# --- The Assembled Sequential Agent ---
github_issue_processing_agent = SequentialAgent(
    name="github_issue_processing_sequential_agent",
    description=(
        "Processes a GitHub issue by fetching its description and providing ADK-specific guidance. "
        'Input should be a JSON string representing the issue details, like \'{"issue_number": "123"}\'. '
        "Output will be a JSON string containing the guidance or an error."
    ),
    # The sequence of agents to run in order.
    sub_agents=[
        get_issue_description_wrapper_agent,
        adk_guidance_wrapper_agent,
        FormatOutputAgent(),
    ],
)
