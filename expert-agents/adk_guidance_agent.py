# expert-agents/adk_guidance_agent.py
import logging
from pydantic import BaseModel

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext # Stays as ReadonlyContext
from google.adk.models import Gemini
from google.genai import types as genai_types

from .context_loader import get_escaped_adk_context_for_llm
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

class AdkGuidanceInput(BaseModel):
    document_text: str

def adk_guidance_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    document_text = ""

    # CORRECTED: Access user_content via _invocation_context
    invocation_ctx = None
    if hasattr(context, '_invocation_context'):
        invocation_ctx = context._invocation_context

    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content and \
       invocation_ctx.user_content.parts:
        first_part_text = invocation_ctx.user_content.parts[0].text
        if first_part_text:
            try:
                input_data = AdkGuidanceInput.model_validate_json(first_part_text)
                document_text = input_data.document_text
                logger.info(f"ADKGuidanceAgent: Received document_text: '{document_text[:200]}...'")
            except Exception as e:
                logger.error(f"ADKGuidanceAgent: Could not parse input: {e}. Content: {first_part_text}")
                document_text = "Error: Could not parse the provided document text. The input from the calling agent might be malformed."
    
    instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 0.5.0.
Your task is to provide guidance and potential solutions based on your comprehensive ADK knowledge and the provided document text.

Your ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---

Provided Document Text:
--- START OF DOCUMENT TEXT ---
{document_text}
--- END OF DOCUMENT TEXT ---

Analyze the "Provided Document Text" in conjunction with "Your ADK Knowledge Context".
Formulate a helpful, detailed, and actionable response based SOLELY on these two pieces of information.
This is your final response to be presented to the user. Do not ask further questions or try to use tools.
"""
    return instruction

adk_guidance_agent = ADKAgent(
    name="adk_guidance_agent",
    description="Provides expert guidance and solutions related to Google ADK based on provided document text and its ADK knowledge.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=adk_guidance_instruction_provider,
    input_schema=AdkGuidanceInput,
    before_model_callback=log_prompt_before_model_call,
    disallow_transfer_to_parent=True, 
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=60000,
        top_p=0.7,
    )
)