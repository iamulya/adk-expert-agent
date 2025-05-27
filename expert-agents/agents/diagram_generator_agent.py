# expert-agents/agents/diagram_generator_agent.py
import logging
import json
from pydantic import BaseModel, Field, ValidationError

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from ..context_loader import get_escaped_adk_context_for_llm # Relative
from ..config import PRO_MODEL_NAME # Relative
from ..callbacks import log_prompt_before_model_call # Relative
from .mermaid_syntax_verifier_agent import mermaid_syntax_verifier_agent # Relative
from ..tools.mermaid_to_png_and_upload_tool import mermaid_gcs_tool_instance, GCS_LINK_STATE_KEY # Relative

logger = logging.getLogger(__name__)

class DiagramGeneratorAgentToolInput(BaseModel):
    diagram_query: str = Field(description="The user's query describing the architecture diagram to be generated.")

def diagram_generator_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_diagram_query = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            try:
                input_data = DiagramGeneratorAgentToolInput.model_validate_json(invocation_ctx.user_content.parts[0].text)
                user_diagram_query = input_data.diagram_query
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(f"DiagramGeneratorAgent: Error parsing input: {e}. Raw input: {invocation_ctx.user_content.parts[0].text[:100]}")
                return "Error: Could not understand the diagram request. Please provide a clear query for the diagram."
    
    if not user_diagram_query:
        return "Error: No diagram query provided. Please specify what kind of diagram you need."

    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    instruction = f"""
You are an AI assistant that generates architecture diagrams in Mermaid syntax. You have access to the ADK (Agent Development Kit) knowledge context.

ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---

The user's request for the diagram is: "{user_diagram_query}"

Follow these steps precisely:
1.  Based on the user's request ("{user_diagram_query}") and the ADK Knowledge Context, generate the Mermaid syntax for the diagram.
    The Mermaid syntax MUST be enclosed in a standard Mermaid code block, like so:
    ```mermaid
    graph TD
        A --> B
    ```
2.  After generating the initial Mermaid syntax, you MUST call the '{mermaid_syntax_verifier_agent.name}' tool.
    The input to this tool MUST be a JSON object with a single key "mermaid_syntax", where the value is the Mermaid syntax you just generated (including the ```mermaid ... ``` block).
    Example call: {{"mermaid_syntax": "```mermaid\ngraph TD; A-->B;\n```"}}
3.  The '{mermaid_syntax_verifier_agent.name}' tool will return the verified (and potentially corrected) Mermaid syntax.
    This response will also be a JSON object, and you need to extract the value of the "mermaid_syntax" key from its "result" field.
4.  Once you have the final, verified Mermaid syntax, you MUST call the '{mermaid_gcs_tool_instance.name}' tool.
    The input to this tool MUST be a JSON object with a single key "mermaid_syntax", where the value is the final verified Mermaid syntax (including the ```mermaid ... ``` block).
5.  Your final response for this turn MUST ONLY be the result from the '{mermaid_gcs_tool_instance.name}' tool (which will be a GCS signed URL or an error message).
    Do not add any conversational fluff, explanations, or greetings around this final URL.
"""
    return instruction

async def diagram_generator_after_agent_cb(callback_context: CallbackContext) -> genai_types.Content | None:
    gcs_link = callback_context.state.get(GCS_LINK_STATE_KEY)
    if gcs_link:
        logger.info(f"DiagramGeneratorAgent (after_agent_callback): Returning GCS link: {gcs_link}")
        return genai_types.Content(parts=[genai_types.Part(text=str(gcs_link))])
    logger.warning("DiagramGeneratorAgent (after_agent_callback): GCS link not found in state.")
    return genai_types.Content(parts=[genai_types.Part(text="Error: Could not generate diagram link.")])

diagram_generator_agent = ADKAgent(
    name="mermaid_diagram_orchestrator_agent",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=diagram_generator_agent_instruction_provider,
    tools=[
        AgentTool(agent=mermaid_syntax_verifier_agent),
        mermaid_gcs_tool_instance,
    ],
    input_schema=DiagramGeneratorAgentToolInput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=diagram_generator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)
