# expert-agents/agents/mermaid_syntax_generator_agent.py
import logging
import json
import re
from pydantic import BaseModel, Field, ValidationError

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.genai import types as genai_types

from ..context_loader import get_escaped_adk_context_for_llm
from ..config import PRO_MODEL_NAME 
from ..callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)

class MermaidSyntaxGeneratorAgentInput(BaseModel):
    diagram_query: str = Field(description="The user's query describing the architecture diagram to be generated.")

def mermaid_syntax_generator_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_diagram_query = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            try:
                input_data = MermaidSyntaxGeneratorAgentInput.model_validate_json(invocation_ctx.user_content.parts[0].text)
                user_diagram_query = input_data.diagram_query
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(f"MermaidSyntaxGeneratorAgent: Error parsing input: {e}. Raw input: {invocation_ctx.user_content.parts[0].text[:100]}")
                return "Error: Could not understand the diagram request for syntax generation. Please provide a clear query."
    
    if not user_diagram_query:
        return "Error: No diagram query provided for syntax generation."

    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    instruction = f"""
You are an AI assistant that specializes in generating Mermaid diagram syntax.
You have access to the ADK (Agent Development Kit) knowledge context.

ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---

The user's request for the diagram is: "{user_diagram_query}"

Your task is to generate the Mermaid syntax for the diagram based on the user's request and the ADK Knowledge Context.
The Mermaid syntax MUST be enclosed in a standard Mermaid code block, like so:
```mermaid
graph TD
    A --> B
    Your output MUST ONLY be the Mermaid code block itself. Do not add any other explanations, greetings, or conversational text.
"""
    return instruction

async def mermaid_syntax_generator_after_agent_cb(callback_context: CallbackContext) -> genai_types.Content | None:
    invocation_ctx = getattr(callback_context, '_invocation_context', None)
    last_event = invocation_ctx.session.events[-1] if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events else None

    mermaid_syntax_output = "Error: Could not extract Mermaid syntax from the agent's response."
    if last_event and last_event.content and last_event.content.parts:
        raw_text_output = last_event.content.parts[0].text
        if raw_text_output:
            try:
                data = json.loads(raw_text_output)
                if isinstance(data, dict) and "result" in data: # ADK might wrap tool-like agent output
                    mermaid_syntax_output = data["result"]
                elif isinstance(data, str): # If JSON loads to a string
                    mermaid_syntax_output = data
                else: # Other JSON structures
                    mermaid_syntax_output = raw_text_output 
            except json.JSONDecodeError:
                mermaid_syntax_output = raw_text_output

    # Ensure it's just the mermaid block or an error message
    if not mermaid_syntax_output.strip().startswith("Error:"):
        match = re.search(r"(```mermaid\s*[\s\S]+?\s*```)", mermaid_syntax_output, re.DOTALL)
        if match:
            mermaid_syntax_output = match.group(1) # Extract the whole block
        else:
            # If no block found and not an error, it's unexpected.
            # We'll pass it along but log a warning.
            logger.warning(f"MermaidSyntaxGeneratorAgent: Output doesn't look like a Mermaid block or an error: {mermaid_syntax_output[:100]}")
            # To be safe, if it's not an error and not a mermaid block, wrap it as if it's raw syntax.
            # This might be too aggressive if the LLM returns a valid non-block syntax.
            # For now, we assume the LLM is instructed to return a block.
            # If it's just raw syntax without the ```mermaid, the PNG tool can handle it.
            # So, if no block is found, we pass the raw_text_output as is.

    logger.info(f"MermaidSyntaxGeneratorAgent (after_agent_callback): Returning Mermaid syntax: {mermaid_syntax_output[:100]}")
    return genai_types.Content(parts=[genai_types.Part(text=mermaid_syntax_output)])

mermaid_syntax_generator_agent = ADKAgent(
    name="mermaid_syntax_generator_agent",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=mermaid_syntax_generator_instruction_provider,
    input_schema=MermaidSyntaxGeneratorAgentInput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=mermaid_syntax_generator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)