"""
This module defines a tool for cleaning text from GitHub issues.

This tool removes common boilerplate phrases from issue templates to provide
a cleaner input for an LLM to analyze.

Note: This tool is part of an alternative, browser-based GitHub issue fetching
flow that is not the primary method used by the agent. The primary flow uses
the `GetGithubIssueDescriptionTool` which gets cleaner data directly from the API.
"""

import logging
import re
from typing import Any, Dict, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from .github_utils import BOILERPLATE_STRINGS_TO_REMOVE

logger = logging.getLogger(__name__)


class CleanGitHubIssueTextToolInput(BaseModel):
    """Input schema for the cleaning tool."""

    raw_text: str = Field(description="The raw text content of the GitHub issue.")


class CleanGitHubIssueTextToolOutput(BaseModel):
    """Output schema for the cleaning tool."""

    cleaned_text: str


class CleanGitHubIssueTextTool(BaseTool):
    """A tool to clean boilerplate and extra newlines from GitHub issue text."""

    def __init__(self):
        super().__init__(
            name="clean_github_issue_text_tool",
            description="Cleans boilerplate and extra newlines from GitHub issue text.",
        )

    @override
    def _get_declaration(self):
        """Defines the tool's interface for the LLM."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=CleanGitHubIssueTextToolInput.model_json_schema(),
        )

    def clean_text_logic(self, text: str) -> str:
        """
        Contains the core logic for cleaning the text.

        Args:
            text: The raw string to be cleaned.

        Returns:
            The cleaned string.
        """
        if not text:
            return ""
        # If the input is already an error message, pass it through without cleaning.
        if (
            text.startswith("Error fetching issue details:")
            or text.startswith("Message from issue fetcher:")
            or text.startswith("Error: No content extracted")
        ):
            return text

        cleaned_text = text
        # Remove all defined boilerplate strings.
        for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
            cleaned_text = cleaned_text.replace(boilerplate, "")

        # Normalize multiple newlines into a maximum of two.
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()

        if not cleaned_text:
            return "GitHub issue content was empty after cleaning."
        return cleaned_text

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Executes the tool's cleaning logic."""
        tool_context.actions.skip_summarization = True
        try:
            input_data = CleanGitHubIssueTextToolInput.model_validate(args)
            cleaned_text = self.clean_text_logic(input_data.raw_text)
            logger.info(
                f"Tool: {self.name} - Cleaned text (first 100 chars): {cleaned_text[:100]}"
            )
            return CleanGitHubIssueTextToolOutput(
                cleaned_text=cleaned_text
            ).model_dump()
        except Exception as e:
            logger.error(
                f"Tool: {self.name} - Error cleaning text: {e}. Args: {args}",
                exc_info=True,
            )
            return CleanGitHubIssueTextToolOutput(
                cleaned_text=f"Error cleaning text: {e}"
            ).model_dump()
