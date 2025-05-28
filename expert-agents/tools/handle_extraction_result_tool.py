# expert-agents/tools/handle_extraction_result_tool.py
import logging
from typing import Any, Dict, Optional, override
from pydantic import BaseModel

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from .extract_github_issue_details_tool import ExtractionResultInput # Import from sibling

logger = logging.getLogger(__name__)

class HandleExtractionResultToolOutput(BaseModel):
    text_content: str

class HandleExtractionResultTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="handle_extraction_result_tool",
            description="Handles the result from webpage extraction, returning details or an error/message string."
        )

    @override
    def _get_declaration(self):
         return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ExtractionResultInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        tool_context.actions.skip_summarization = True
        try:
            input_data = ExtractionResultInput.model_validate(args)
            if input_data.extracted_details:
                logger.info(f"Tool: {self.name} - Extraction successful.")
                return HandleExtractionResultToolOutput(text_content=input_data.extracted_details).model_dump()
            elif input_data.error:
                logger.error(f"Tool: {self.name} - Extraction failed with error: {input_data.error}")
                return HandleExtractionResultToolOutput(text_content=f"Error fetching issue details: {input_data.error}").model_dump()
            elif input_data.message:
                logger.info(f"Tool: {self.name} - Extraction returned a message: {input_data.message}")
                return HandleExtractionResultToolOutput(text_content=f"Message from issue fetcher: {input_data.message}").model_dump()
            else:
                logger.warning(f"Tool: {self.name} - Extraction returned no details, error, or message.")
                return HandleExtractionResultToolOutput(text_content="Error: No content extracted from the GitHub issue page.").model_dump()
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error processing extraction result: {e}. Args: {args}", exc_info=True)
            return HandleExtractionResultToolOutput(text_content=f"Error processing extraction data: {e}").model_dump()
