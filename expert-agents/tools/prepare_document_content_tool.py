# expert-agents/tools/prepare_document_content_tool.py
import logging
from typing import Any, Dict, Literal, override
from pydantic import BaseModel, Field, ValidationError

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

class PrepareDocumentContentToolInput(BaseModel):
    markdown_content: str = Field(description="The generated Markdown content for the document.")
    document_type: Literal["pdf", "html", "pptx"] = Field(description="The type of document requested by the user (pdf, html, or pptx).")
    output_filename_base: str = Field(description="A base name for the output file, e.g., 'adk_report'. The document_generator_agent will append the correct extension.")
    original_user_query: str = Field(description="The original user query that requested the document.")

class PrepareDocumentContentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="prepare_document_content_tool",
            description="Gathers generated markdown content, document type, and filename base. This tool is called by the orchestrator agent; its output triggers the actual document generation agent."
        )

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=PrepareDocumentContentToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        logger.info(f"PrepareDocumentContentTool 'run_async' called with args: {str(args)[:200]}. Returning these args directly.")
        try:
            validated_args = PrepareDocumentContentToolInput.model_validate(args)
            return validated_args.model_dump()
        except ValidationError as ve:
            logger.error(f"PrepareDocumentContentTool: LLM provided invalid arguments: {ve}. Args: {args}", exc_info=True)
            return {"error": f"Invalid arguments from LLM for content preparation: {str(ve)}", "original_args": args}
