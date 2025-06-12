"""
This module defines the Mermaid Diagram Orchestrator Agent.

This agent manages the end-to-end process of creating a Mermaid diagram from a
user's request. It operates in a two-step process:

1.  **Generate Syntax**: It first calls the `mermaid_syntax_generator_agent` to
    convert the user's natural language query (e.g., "draw a diagram of the
    login flow") into Mermaid diagram syntax.
2.  **Generate Image**: It then takes the generated syntax and passes it to the
    `MermaidToPngAndUploadTool`, which uses the `mmdc` command-line tool to
    convert the syntax into a PNG image, upload it to GCS, and return a link.

This orchestration ensures a clear separation of concerns between generating the
syntax (an LLM task) and converting it to an image (a deterministic tool task).
"""

import logging
import json
from pydantic import BaseModel, Field, ValidationError

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from ..config import DEFAULT_MODEL_NAME
from ..callbacks import log_prompt_before_model_call
from .mermaid_syntax_generator_agent import (
    mermaid_syntax_generator_agent,
    MermaidSyntaxGeneratorAgentInput,
)
from ..tools.mermaid_to_png_and_upload_tool import (
    mermaid_gcs_tool_instance,
    GCS_LINK_STATE_KEY,
)

logger = logging.getLogger(__name__)


class DiagramGeneratorAgentToolInput(BaseModel):  # Input from root_agent
    """
    Defines the input schema for this agent when it is called as a tool by the
    root_agent.
    """

    diagram_query: str = Field(
        description="The user's query describing the architecture diagram to be generated."
    )


def diagram_orchestrator_instruction_provider(context: ReadonlyContext) -> str:
    """
    Dynamically generates the instruction for the orchestrator agent's LLM.

    This logic checks the conversation history to determine which step of the
    orchestration is next. If Mermaid syntax has already been generated, it
    instructs the agent to call the image generation tool. Otherwise, it
    instructs the agent to call the syntax generation agent first.
    """
    invocation_ctx = getattr(context, "_invocation_context", None)
    user_diagram_query = ""

    # Parse the initial request from the root agent to get the user's query.
    if (
        invocation_ctx
        and hasattr(invocation_ctx, "user_content")
        and invocation_ctx.user_content
    ):
        if (
            invocation_ctx.user_content.parts
            and invocation_ctx.user_content.parts[0].text
        ):
            try:
                input_data = DiagramGeneratorAgentToolInput.model_validate_json(
                    invocation_ctx.user_content.parts[0].text
                )
                user_diagram_query = input_data.diagram_query
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(
                    f"DiagramOrchestratorAgent: Error parsing input: {e}. Raw input: {invocation_ctx.user_content.parts[0].text[:100]}"
                )
                return (
                    "Error: Could not understand the diagram request for orchestration."
                )

    if not user_diagram_query:
        return "Error: No diagram query provided for orchestration."

    mermaid_syntax_from_previous_step = None
    # Check if the last event was a response from the mermaid_syntax_generator_agent.
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events:
        last_event = invocation_ctx.session.events[-1]
        if (
            last_event.author == mermaid_diagram_orchestrator_agent.name
            and last_event.content
            and last_event.content.parts
            and last_event.content.parts[0].function_response
            and last_event.content.parts[0].function_response.name
            == mermaid_syntax_generator_agent.name
        ):
            tool_output_container = last_event.content.parts[
                0
            ].function_response.response
            # The mermaid_syntax_generator_agent's after_agent_cb ensures its output is the direct mermaid syntax string.
            # When called as an AgentTool, ADK wraps this in {'result': ...}.
            if (
                isinstance(tool_output_container, dict)
                and "result" in tool_output_container
            ):
                mermaid_syntax_from_previous_step = tool_output_container["result"]
            elif isinstance(
                tool_output_container, str
            ):  # Should not happen if called as AgentTool but handle for safety
                mermaid_syntax_from_previous_step = tool_output_container
            else:
                logger.error(
                    f"DiagramOrchestrator: Unexpected output format from syntax generator: {tool_output_container}"
                )
                return "Error: Failed to get Mermaid syntax from the generator agent due to unexpected format."

            # If the syntax generator itself returned an error, propagate it immediately.
            if (
                mermaid_syntax_from_previous_step
                and mermaid_syntax_from_previous_step.strip().startswith("Error:")
            ):
                logger.error(
                    f"DiagramOrchestrator: Syntax generator returned an error: {mermaid_syntax_from_previous_step}"
                )
                # This error message will be the final output of this agent for this turn.
                return mermaid_syntax_from_previous_step

    if mermaid_syntax_from_previous_step:
        # If we have syntax, instruct the LLM to call the PNG generation tool.
        logger.info(
            "DiagramOrchestratorAgent: Received Mermaid syntax. Instructing to call mermaid_gcs_tool_instance."
        )
        tool_input_payload = json.dumps(
            {"mermaid_syntax": mermaid_syntax_from_previous_step}
        )
        instruction = f"""
You have received Mermaid syntax.
Your task is to call the '{mermaid_gcs_tool_instance.name}' tool with this syntax.
The input to this tool MUST be the following JSON object:
{tool_input_payload}
Your final response for this turn MUST ONLY be the result from the '{mermaid_gcs_tool_instance.name}' tool.
Do not add any conversational fluff, explanations, or greetings.
"""
    else:
        # This is the first step: instruct the LLM to call the syntax generator agent.
        logger.info(
            "DiagramOrchestratorAgent: Instructing to call mermaid_syntax_generator_agent."
        )
        syntax_agent_input_payload = MermaidSyntaxGeneratorAgentInput(
            diagram_query=user_diagram_query
        ).model_dump_json()
        instruction = f"""
You are an orchestrator for generating Mermaid diagrams.
The user's request for a diagram is: "{user_diagram_query}"

Your first step is to call the '{mermaid_syntax_generator_agent.name}' tool to generate the Mermaid syntax.
The input to this tool MUST be the following JSON string:
{syntax_agent_input_payload}
Output only this tool call. Do not add any other text.
"""
    return instruction


async def diagram_orchestrator_after_agent_cb(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """
    Callback to format the final output of the orchestration.

    This callback inspects the last event to find the result from the final tool
    call (either the PNG tool or an error from the syntax generator). It also
    checks the tool context state, where the PNG tool writes its final URL.
    This result becomes the definitive output of this agent.
    """
    final_output_text = "Error: Could not determine the final output from the diagram generation process."
    invocation_ctx = getattr(callback_context, "_invocation_context", None)
    last_event = (
        invocation_ctx.session.events[-1]
        if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events
        else None
    )

    # The GCS tool writes its definitive output to the state. This is the most reliable source.
    if (
        last_event
        and last_event.content
        and last_event.content.parts
        and last_event.content.parts[0].function_response
    ):
        function_response_part = last_event.content.parts[0].function_response
        tool_name_called = function_response_part.name

        if tool_name_called == mermaid_gcs_tool_instance.name:
            # The mermaid_gcs_tool_instance writes its definitive output to GCS_LINK_STATE_KEY.
            state_output = callback_context.state.get(GCS_LINK_STATE_KEY)
            if state_output:
                final_output_text = str(state_output)
                logger.info(
                    f"DiagramOrchestrator: Got output from state for GCS tool: {final_output_text[:100]}"
                )
            else:
                # Fallback to direct response if state key is missing (should be rare).
                response_data = function_response_part.response
                if isinstance(response_data, dict) and "result" in response_data:
                    final_output_text = str(response_data["result"])
                elif isinstance(response_data, str):
                    final_output_text = response_data
                else:
                    final_output_text = f"Error: Unexpected response format from {mermaid_gcs_tool_instance.name} and state key missing."
                logger.warning(
                    f"DiagramOrchestrator: GCS_LINK_STATE_KEY not found, used direct response: {final_output_text[:100]}"
                )

        elif tool_name_called == mermaid_syntax_generator_agent.name:
            # This case occurs if the syntax generator was the last tool called, which
            # usually means it errored out and the GCS tool was never reached.
            # Its after_agent_cb ensures its output (error or syntax) is in response['result'].
            response_data = function_response_part.response
            if isinstance(response_data, dict) and "result" in response_data:
                final_output_text = response_data["result"]
            elif isinstance(
                response_data, str
            ):  # Should be wrapped if called as AgentTool
                final_output_text = response_data
            else:
                final_output_text = f"Error: Unexpected response format from {mermaid_syntax_generator_agent.name}."
            logger.info(
                f"DiagramOrchestrator: Output from syntax generator (likely an error if it's the last step): {final_output_text[:100]}"
            )
        else:
            # Handle unexpected tool calls.
            logger.warning(
                f"DiagramOrchestrator (after_agent_callback): Last tool call was '{tool_name_called}', not GCS or syntax generator."
            )
            response_data = function_response_part.response  # Fallback
            if isinstance(response_data, str):
                final_output_text = response_data
            elif isinstance(response_data, dict) and "result" in response_data:
                final_output_text = str(response_data["result"])
            else:
                final_output_text = "Error: Unknown tool response as last event."
    elif (
        last_event
        and last_event.content
        and last_event.content.parts
        and last_event.content.parts[0].text
    ):
        # This case handles if the orchestrator's LLM directly responds with text
        # (e.g., with an error message generated by the instruction_provider).
        final_output_text = last_event.content.parts[0].text
        logger.info(
            f"DiagramOrchestrator: Last event was a direct text response: {final_output_text[:100]}"
        )
    else:
        logger.error(
            "DiagramOrchestrator: No relevant last event or function response found to determine final output."
        )

    logger.info(
        f"DiagramOrchestratorAgent (after_agent_callback): Final output being returned: {final_output_text[:200]}"
    )
    return genai_types.Content(parts=[genai_types.Part(text=final_output_text)])


# The agent definition.
mermaid_diagram_orchestrator_agent = ADKAgent(
    name="mermaid_diagram_orchestrator_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=diagram_orchestrator_instruction_provider,
    tools=[
        # This agent can call another agent (the syntax generator) as a tool.
        AgentTool(agent=mermaid_syntax_generator_agent),
        # It can also call a regular tool (for image generation).
        mermaid_gcs_tool_instance,
    ],
    # This schema defines how this agent is called by the root agent.
    input_schema=DiagramGeneratorAgentToolInput,
    # This agent is a specialist and should not delegate work upwards or sideways.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=diagram_orchestrator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0),
)
