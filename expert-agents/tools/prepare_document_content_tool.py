"""
This module defines the Prepare Document Content Tool.

This tool plays a crucial role in the two-step document generation process.
It doesn't perform any complex logic itself. Instead, it acts as a structured
data container or a "staging" tool.

The root agent's LLM is prompted to first generate Markdown content and then
call this tool, passing the generated content, the desired document type, and a
filename. The root agent's `after_tool_callback` then detects the output from
this tool, which triggers the next step: calling the `document_generator_agent`
with this structured data.

This pattern enforces a clean separation between content generation (a creative
LLM task) and file generation (a deterministic tool task), improving reliability.
"""

import logging
from typing import Any, Dict, Literal, override
from pydantic import BaseModel, Field, ValidationError

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class PrepareDocumentContentToolInput(BaseModel):
    """Input schema for the content preparation tool."""

    markdown_content: str = Field(
        description="The generated Markdown content for the document."
    )
    document_type: Literal["pdf", "html", "pptx"] = Field(
        description="The type of document requested by the user (pdf, html, or pptx)."
    )
    output_filename_base: str = Field(
        description="A base name for the output file, e.g., 'adk_report'. The document_generator_agent will append the correct extension."
    )
    original_user_query: str = Field(
        description="The original user query that requested the document."
    )


class PrepareDocumentContentTool(BaseTool):
    """A tool to gather and stage content for the final document generation step."""

    def __init__(self):
        super().__init__(
            name="prepare_document_content_tool",
            description="Gathers generated markdown content, document type, and filename base. This tool is called by the orchestrator agent; its output triggers the actual document generation agent.",
        )

    @override
    def _get_declaration(self):
        """Defines the tool's interface for the LLM."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=PrepareDocumentContentToolInput.model_json_schema(),
        )

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> Dict[str, Any]:
        """
        Executes the tool's "logic", which is simply to validate and pass through arguments.

        This tool acts as a data shuttle. It takes the arguments provided by the
        LLM, validates them against its Pydantic schema, and returns them as a
        dictionary. This dictionary then becomes the tool's output, which the
        root agent's `after_tool_callback` will process to initiate the next
        step of the workflow.

        Args:
            args: A dictionary of arguments from the LLM.
            tool_context: The context of the tool call.

        Returns:
            A dictionary of the validated arguments, or an error dictionary if
            validation fails.
        """
        logger.info(
            f"PrepareDocumentContentTool 'run_async' called with args: {str(args)[:200]}. Returning these args directly."
        )
        try:
            # Validate that the arguments provided by the LLM match the expected schema.
            validated_args = PrepareDocumentContentToolInput.model_validate(args)
            return validated_args.model_dump()
        except ValidationError as ve:
            # If the LLM failed to provide the correct arguments, log the error
            # and return an error dictionary for debugging.
            logger.error(
                f"PrepareDocumentContentTool: LLM provided invalid arguments: {ve}. Args: {args}",
                exc_info=True,
            )
            return {
                "error": f"Invalid arguments from LLM for content preparation: {str(ve)}",
                "original_args": args,
            }
