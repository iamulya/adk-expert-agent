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
from .mermaid_syntax_generator_agent import mermaid_syntax_generator_agent, MermaidSyntaxGeneratorAgentInput
from ..tools.mermaid_to_png_and_upload_tool import mermaid_gcs_tool_instance, GCS_LINK_STATE_KEY

logger = logging.getLogger(__name__)

class DiagramGeneratorAgentToolInput(BaseModel): # Input from root_agent
    diagram_query: str = Field(description="The user's query describing the architecture diagram to be generated.")

def diagram_orchestrator_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_diagram_query = ""

    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            try:
                input_data = DiagramGeneratorAgentToolInput.model_validate_json(invocation_ctx.user_content.parts[0].text)
                user_diagram_query = input_data.diagram_query
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(f"DiagramOrchestratorAgent: Error parsing input: {e}. Raw input: {invocation_ctx.user_content.parts[0].text[:100]}")
                return "Error: Could not understand the diagram request for orchestration."
    
    if not user_diagram_query:
        return "Error: No diagram query provided for orchestration."

    mermaid_syntax_from_previous_step = None
    # Check if the last event was a response from the mermaid_syntax_generator_agent
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events:
        last_event = invocation_ctx.session.events[-1]
        if last_event.author == mermaid_diagram_orchestrator_agent.name and \
           last_event.content and last_event.content.parts and \
           last_event.content.parts[0].function_response and \
           last_event.content.parts[0].function_response.name == mermaid_syntax_generator_agent.name:
            
            tool_output_container = last_event.content.parts[0].function_response.response
            # The mermaid_syntax_generator_agent's after_agent_cb ensures its output is the direct mermaid syntax string.
            # When called as an AgentTool, ADK wraps this in {'result': ...}
            if isinstance(tool_output_container, dict) and "result" in tool_output_container:
                 mermaid_syntax_from_previous_step = tool_output_container["result"]
            elif isinstance(tool_output_container, str): # Should not happen if called as AgentTool
                 mermaid_syntax_from_previous_step = tool_output_container
            else:
                logger.error(f"DiagramOrchestrator: Unexpected output format from syntax generator: {tool_output_container}")
                return "Error: Failed to get Mermaid syntax from the generator agent due to unexpected format."

            # If the syntax generator itself returned an error, propagate it.
            if mermaid_syntax_from_previous_step and mermaid_syntax_from_previous_step.strip().startswith("Error:"):
                logger.error(f"DiagramOrchestrator: Syntax generator returned an error: {mermaid_syntax_from_previous_step}")
                # This error message will be the final output of this agent for this turn.
                return mermaid_syntax_from_previous_step 

    if mermaid_syntax_from_previous_step:
        logger.info("DiagramOrchestratorAgent: Received Mermaid syntax. Instructing to call mermaid_gcs_tool_instance.")
        tool_input_payload = json.dumps({"mermaid_syntax": mermaid_syntax_from_previous_step})
        instruction = f"""
You have received Mermaid syntax.
Your task is to call the '{mermaid_gcs_tool_instance.name}' tool with this syntax.
The input to this tool MUST be the following JSON object:
{tool_input_payload}
Your final response for this turn MUST ONLY be the result from the '{mermaid_gcs_tool_instance.name}' tool.
Do not add any conversational fluff, explanations, or greetings.
"""
    else:
        # This is the first step: call the syntax generator.
        logger.info("DiagramOrchestratorAgent: Instructing to call mermaid_syntax_generator_agent.")
        syntax_agent_input_payload = MermaidSyntaxGeneratorAgentInput(diagram_query=user_diagram_query).model_dump_json()
        instruction = f"""
You are an orchestrator for generating Mermaid diagrams.
The user's request for a diagram is: "{user_diagram_query}"

Your first step is to call the '{mermaid_syntax_generator_agent.name}' tool to generate the Mermaid syntax.
The input to this tool MUST be the following JSON string:
{syntax_agent_input_payload}
Output only this tool call. Do not add any other text.
"""
    return instruction

async def diagram_orchestrator_after_agent_cb(callback_context: CallbackContext) -> genai_types.Content | None:
    final_output_text = "Error: Could not determine the final output from the diagram generation process."
    invocation_ctx = getattr(callback_context, '_invocation_context', None)
    last_event = invocation_ctx.session.events[-1] if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events else None

    if last_event and last_event.content and last_event.content.parts and \
       last_event.content.parts[0].function_response:
        
        function_response_part = last_event.content.parts[0].function_response
        tool_name_called = function_response_part.name
        
        if tool_name_called == mermaid_gcs_tool_instance.name:
            # The mermaid_gcs_tool_instance writes its definitive output to GCS_LINK_STATE_KEY.
            state_output = callback_context.state.get(GCS_LINK_STATE_KEY)
            if state_output:
                final_output_text = str(state_output)
                logger.info(f"DiagramOrchestrator: Got output from state for GCS tool: {final_output_text[:100]}")
            else: 
                # Fallback to direct response if state key is missing (should be rare)
                response_data = function_response_part.response
                if isinstance(response_data, dict) and 'result' in response_data:
                    final_output_text = str(response_data['result'])
                elif isinstance(response_data, str):
                    final_output_text = response_data
                else:
                    final_output_text = f"Error: Unexpected response format from {mermaid_gcs_tool_instance.name} and state key missing."
                logger.warning(f"DiagramOrchestrator: GCS_LINK_STATE_KEY not found, used direct response: {final_output_text[:100]}")

        elif tool_name_called == mermaid_syntax_generator_agent.name:
            # If the last tool called was the syntax generator, it means the GCS tool wasn't called.
            # This typically happens if the syntax generator itself errored.
            # Its after_agent_cb ensures its output (error or syntax) is in response['result'].
            response_data = function_response_part.response
            if isinstance(response_data, dict) and "result" in response_data:
                final_output_text = response_data["result"]
            elif isinstance(response_data, str): # Should be wrapped if called as AgentTool
                final_output_text = response_data
            else:
                final_output_text = f"Error: Unexpected response format from {mermaid_syntax_generator_agent.name}."
            logger.info(f"DiagramOrchestrator: Output from syntax generator (likely an error if it's the last step): {final_output_text[:100]}")
        else:
            logger.warning(f"DiagramOrchestrator (after_agent_callback): Last tool call was '{tool_name_called}', not GCS or syntax generator.")
            response_data = function_response_part.response # Fallback
            if isinstance(response_data, str): final_output_text = response_data
            elif isinstance(response_data, dict) and 'result' in response_data : final_output_text = str(response_data['result'])
            else: final_output_text = "Error: Unknown tool response as last event."
    elif last_event and last_event.content and last_event.content.parts and last_event.content.parts[0].text:
        # This case handles if the orchestrator's LLM directly responds (e.g., with an error message from instruction_provider).
        final_output_text = last_event.content.parts[0].text
        logger.info(f"DiagramOrchestrator: Last event was a direct text response: {final_output_text[:100]}")
    else:
        logger.error("DiagramOrchestrator: No relevant last event or function response found to determine final output.")

    logger.info(f"DiagramOrchestratorAgent (after_agent_callback): Final output being returned: {final_output_text[:200]}")
    return genai_types.Content(parts=[genai_types.Part(text=final_output_text)])

mermaid_diagram_orchestrator_agent = ADKAgent(
    name="mermaid_diagram_orchestrator_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=diagram_orchestrator_instruction_provider,
    tools=[
        AgentTool(agent=mermaid_syntax_generator_agent), # New agent for syntax generation
        mermaid_gcs_tool_instance,
    ],
    input_schema=DiagramGeneratorAgentToolInput, # Schema for when this agent is called as a tool by root_agent
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=diagram_orchestrator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)