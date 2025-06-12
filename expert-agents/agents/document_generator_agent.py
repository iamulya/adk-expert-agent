"""
This module defines the Document Generator Agent.

This is a specialized agent responsible for the final step of document creation.
It takes pre-generated Markdown content and a desired document type, then uses
its internal tools (which wrap the `marp-cli`) to generate the final file
(PDF, HTML, or PPTX), upload it to GCS, and return a link. It is designed to
be called as a tool by the root orchestrator agent.
"""

import logging
from typing import Literal

from google.adk.agents import Agent as LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from ..tools import (
    DOC_LINK_STATE_KEY,
    generate_pdf_from_markdown_with_gcs,
    generate_html_slides_from_markdown_with_gcs,
    generate_pptx_slides_from_markdown_with_gcs,
)
from ..config import PRO_MODEL_NAME
from ..callbacks import log_prompt_before_model_call

logger = logging.getLogger(__name__)


# --- Pydantic Models for this Agent's internal logic ---
class DocumentGenerationInput(BaseModel):
    """Defines the input arguments for the internal document generation tools."""

    markdown_content: str = Field(description="The Markdown content for the document.")
    output_filename: str = Field(
        description="The desired output filename (e.g., 'report.pdf', 'slides.html')."
    )


# --- Document Generator Agent's Internal Tools (wrapped FunctionTools) ---
# These tools are used *by* the document_generator_agent's LLM to perform the
# actual file generation by calling the underlying marp-cli functions.

_marp_pdf_tool_gcs = FunctionTool(func=generate_pdf_from_markdown_with_gcs)
_marp_html_tool_gcs = FunctionTool(func=generate_html_slides_from_markdown_with_gcs)
_marp_pptx_tool_gcs = FunctionTool(func=generate_pptx_slides_from_markdown_with_gcs)


# --- Pydantic Model for the input TO this document_generator_agent WHEN CALLED AS A TOOL ---
# This is what the root_agent will provide.
class DocumentGeneratorAgentToolInput(BaseModel):
    """
    Defines the input schema for this agent when it is called as a tool by
    another agent (e.g., the root_agent).
    """

    markdown_content: str = Field(
        description="The complete Markdown content for the document."
    )
    document_type: Literal["pdf", "html", "pptx"] = Field(
        description="The type of document to generate: 'pdf', 'html' (for slides), or 'pptx'."
    )
    output_filename: str = Field(
        description="The desired base filename (e.g., 'my_report', 'presentation_slides'). The agent will ensure the correct extension."
    )


def document_generator_instruction_provider(context: ReadonlyContext) -> str:
    """
    Generates the instruction for the document_generator_agent's LLM.

    It receives markdown, document_type, and a base filename from the root agent.
    Its job is to parse this input and then construct a new prompt that
    forces the LLM to call the correct internal Marp+GCS tool (`_marp_pdf_tool_gcs`, etc.)
    with the correct arguments.
    """
    invocation_ctx = getattr(context, "_invocation_context", None)
    input_json_str = ""
    if (
        invocation_ctx
        and hasattr(invocation_ctx, "user_content")
        and invocation_ctx.user_content
    ):
        if (
            invocation_ctx.user_content.parts
            and invocation_ctx.user_content.parts[0].text
        ):
            input_json_str = invocation_ctx.user_content.parts[0].text

    logger.info(
        f"DocumentGeneratorAgent: Received input JSON string: {input_json_str[:200]}"
    )

    try:
        # This agent expects its input (when called as a tool) to be a JSON string
        # conforming to DocumentGeneratorAgentToolInput.
        parsed_input = DocumentGeneratorAgentToolInput.model_validate_json(
            input_json_str
        )

        markdown_content = parsed_input.markdown_content
        doc_type = parsed_input.document_type
        base_filename = parsed_input.output_filename

        tool_to_call = ""
        final_filename = ""

        # Based on the requested document type, select the correct internal tool to call
        # and construct the final filename with the proper extension.
        if doc_type == "pdf":
            tool_to_call = _marp_pdf_tool_gcs.name
            final_filename = (
                base_filename
                if base_filename.endswith(".pdf")
                else base_filename + ".pdf"
            )
        elif doc_type == "html":
            tool_to_call = _marp_html_tool_gcs.name
            final_filename = (
                base_filename
                if base_filename.endswith(".html")
                else base_filename + ".html"
            )
        elif doc_type == "pptx":
            tool_to_call = _marp_pptx_tool_gcs.name
            final_filename = (
                base_filename
                if base_filename.endswith(".pptx")
                else base_filename + ".pptx"
            )
        else:
            # This case should ideally not be reached due to Pydantic validation on the input model.
            return "Error: Invalid document_type specified. Must be 'pdf', 'html', or 'pptx'."

        # Prepare the arguments for the selected internal tool.
        tool_args = DocumentGenerationInput(
            markdown_content=markdown_content, output_filename=final_filename
        ).model_dump_json()

        logger.info(
            f"DocumentGeneratorAgent: Instructing to call '{tool_to_call}' with filename '{final_filename}'"
        )
        # Return a highly specific instruction to force the LLM to call the desired tool and nothing else.
        return f"You are a document generation specialist. Your task is to call the tool '{tool_to_call}' with the following JSON arguments: {tool_args}. Output only the tool call. Do not add any other text."

    except Exception as e:
        logger.error(
            f"DocumentGeneratorAgent: Error processing input '{input_json_str[:200]}': {e}",
            exc_info=True,
        )
        return f"Error: Could not parse the input for document generation. Expected JSON with 'markdown_content', 'document_type', 'output_filename'. Received: {input_json_str[:100]}. Error: {e}"


async def document_generator_after_agent_cb(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """
    This callback runs after this agent has finished its internal processing.

    It retrieves the final result (the GCS link or an error message) which was
    stored in the context's state by the internal marp tool. This result is then
    returned as the agent's definitive final output, which will be passed back
    to the root agent.
    """
    gcs_link = callback_context.state.get(DOC_LINK_STATE_KEY)
    if gcs_link:
        logger.info(
            f"DocumentGeneratorAgent (after_agent_callback): Returning GCS link: {gcs_link}"
        )
        # This Content object becomes the final output of the DocumentGeneratorAgent.
        return genai_types.Content(parts=[genai_types.Part(text=str(gcs_link))])
    logger.warning(
        "DocumentGeneratorAgent (after_agent_callback): GCS link not found in state."
    )
    return genai_types.Content(
        parts=[genai_types.Part(text="Error: Could not generate diagram link.")]
    )


# The agent definition.
document_generator_agent = LlmAgent(
    name="document_creation_specialist_agent",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=document_generator_instruction_provider,
    tools=[
        _marp_pdf_tool_gcs,
        _marp_html_tool_gcs,
        _marp_pptx_tool_gcs,
    ],
    # This agent is a specialist that performs a single task; it should not delegate.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    # This schema defines how other agents (like the root_agent) should call this agent.
    input_schema=DocumentGeneratorAgentToolInput,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=document_generator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0),
)
