"""
This module defines a tool to handle the output of the web extraction tool.

Note: This tool is part of an alternative, browser-based GitHub issue fetching
flow that is not the primary method used by the agent. Its purpose is to take
the structured output (details, error, or message) from the
`ExtractGitHubIssueDetailsTool` and flatten it into a single string for the
next tool in the chain.
"""

import logging
from typing import Any, Dict, override
from pydantic import BaseModel

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from .extract_github_issue_details_tool import (
    ExtractionResultInput,
)  # Import from sibling

logger = logging.getLogger(__name__)


class HandleExtractionResultToolOutput(BaseModel):
    """Output schema for the result handling tool."""

    text_content: str


class HandleExtractionResultTool(BaseTool):
    """A tool to process the result from webpage extraction, returning a single string."""

    def __init__(self):
        super().__init__(
            name="handle_extraction_result_tool",
            description="Handles the result from webpage extraction, returning details or an error/message string.",
        )

    @override
    def _get_declaration(self):
        """Defines the tool's interface for the LLM."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            # The input for this tool is the output of the previous tool.
            parameters=ExtractionResultInput.model_json_schema(),
        )

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Executes the tool's result handling logic."""
        tool_context.actions.skip_summarization = True
        try:
            input_data = ExtractionResultInput.model_validate(args)
            # Prioritize returning the actual details if available.
            if input_data.extracted_details:
                logger.info(f"Tool: {self.name} - Extraction successful.")
                return HandleExtractionResultToolOutput(
                    text_content=input_data.extracted_details
                ).model_dump()
            # If there was an error, format it as an error string.
            elif input_data.error:
                logger.error(
                    f"Tool: {self.name} - Extraction failed with error: {input_data.error}"
                )
                return HandleExtractionResultToolOutput(
                    text_content=f"Error fetching issue details: {input_data.error}"
                ).model_dump()
            # If there was a message, format it as a message string.
            elif input_data.message:
                logger.info(
                    f"Tool: {self.name} - Extraction returned a message: {input_data.message}"
                )
                return HandleExtractionResultToolOutput(
                    text_content=f"Message from issue fetcher: {input_data.message}"
                ).model_dump()
            # Fallback case.
            else:
                logger.warning(
                    f"Tool: {self.name} - Extraction returned no details, error, or message."
                )
                return HandleExtractionResultToolOutput(
                    text_content="Error: No content extracted from the GitHub issue page."
                ).model_dump()
        except Exception as e:
            logger.error(
                f"Tool: {self.name} - Error processing extraction result: {e}. Args: {args}",
                exc_info=True,
            )
            return HandleExtractionResultToolOutput(
                text_content=f"Error processing extraction data: {e}"
            ).model_dump()
