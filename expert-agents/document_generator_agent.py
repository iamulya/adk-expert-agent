# expert-agents/document_generator.py
import os
import logging
import json
from typing import Literal, Dict, Any

from google.adk.agents import Agent as LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from .tools import (
    generate_pdf_from_markdown_with_gcs,
    generate_html_slides_from_markdown_with_gcs,
    generate_pptx_slides_from_markdown_with_gcs
)
from .config import DEFAULT_MODEL_NAME, PRO_MODEL_NAME
from .callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)

# --- Pydantic Models for Document Generator Agent's internal tools ---
class DocumentGenerationInput(BaseModel):
    markdown_content: str = Field(description="The Markdown content for the document.")
    output_filename: str = Field(description="The desired output filename (e.g., 'report.pdf', 'slides.html').")

# --- Document Generator Agent's Internal Tools (wrapped FunctionTools) ---
# These tools are used *by* the document_generator_agent's LLM.

_marp_pdf_tool_gcs = FunctionTool(func=generate_pdf_from_markdown_with_gcs)
_marp_html_tool_gcs = FunctionTool(func=generate_html_slides_from_markdown_with_gcs)
_marp_pptx_tool_gcs = FunctionTool(func=generate_pptx_slides_from_markdown_with_gcs)


# --- Pydantic Model for the input TO this document_generator_agent WHEN CALLED AS A TOOL ---
# This is what the root_agent will provide.
class DocumentGeneratorAgentToolInput(BaseModel):
    markdown_content: str = Field(description="The complete Markdown content for the document.")
    document_type: Literal["pdf", "html", "pptx"] = Field(description="The type of document to generate: 'pdf', 'html' (for slides), or 'pptx'.")
    output_filename: str = Field(description="The desired base filename (e.g., 'my_report', 'presentation_slides'). The agent will ensure the correct extension.")

def document_generator_instruction_provider(context: ReadonlyContext) -> str:
    """
    Instruction for the document_generator_agent.
    It receives markdown, document_type, and a base filename.
    Its job is to call the correct internal marp+gcs tool.
    """
    invocation_ctx = getattr(context, '_invocation_context', None)
    input_json_str = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            input_json_str = invocation_ctx.user_content.parts[0].text
    
    logger.info(f"DocumentGeneratorAgent: Received input JSON string: {input_json_str[:200]}")

    try:
        # This agent expects its input (when called as a tool) to be a JSON string
        # conforming to DocumentGeneratorAgentToolInput.
        parsed_input = DocumentGeneratorAgentToolInput.model_validate_json(input_json_str)
        
        markdown_content = parsed_input.markdown_content
        doc_type = parsed_input.document_type
        base_filename = parsed_input.output_filename

        tool_to_call = ""
        final_filename = ""

        if doc_type == "pdf":
            tool_to_call = _marp_pdf_tool_gcs.name
            final_filename = base_filename if base_filename.endswith(".pdf") else base_filename + ".pdf"
        elif doc_type == "html":
            tool_to_call = _marp_html_tool_gcs.name
            final_filename = base_filename if base_filename.endswith(".html") else base_filename + ".html"
        elif doc_type == "pptx":
            tool_to_call = _marp_pptx_tool_gcs.name
            final_filename = base_filename if base_filename.endswith(".pptx") else base_filename + ".pptx"
        else:
            return "Error: Invalid document_type specified. Must be 'pdf', 'html', or 'pptx'."

        tool_args = DocumentGenerationInput(
            markdown_content=markdown_content, 
            output_filename=final_filename
        ).model_dump_json()

        logger.info(f"DocumentGeneratorAgent: Instructing to call '{tool_to_call}' with filename '{final_filename}'")
        return f"You are a document generation specialist. Your task is to call the tool '{tool_to_call}' with the following JSON arguments: {tool_args}. Output only the tool call. Do not add any other text."

    except Exception as e:
        logger.error(f"DocumentGeneratorAgent: Error processing input '{input_json_str[:200]}': {e}", exc_info=True)
        return f"Error: Could not parse the input for document generation. Expected JSON with 'markdown_content', 'document_type', 'output_filename'. Received: {input_json_str[:100]}. Error: {e}"


document_generator_agent = LlmAgent(
    name="document_creation_specialist_agent",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=document_generator_instruction_provider,
    tools=[
        _marp_pdf_tool_gcs,
        _marp_html_tool_gcs,
        _marp_pptx_tool_gcs,
    ],
    # This agent is a specialist, does one thing.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    input_schema=DocumentGeneratorAgentToolInput, # Schema for when *this agent* is called as a tool
    before_model_callback=log_prompt_before_model_call,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)