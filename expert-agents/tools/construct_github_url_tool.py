"""
This module defines a simple tool to construct a GitHub issue URL.

Note: This tool is part of an alternative, browser-based GitHub issue fetching
flow that is not the primary method used by the agent. It would be used to
generate the URL to pass to a browser-based tool like `ExtractGitHubIssueDetailsTool`.
"""

import logging
from typing import Any, Dict, Optional, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class ConstructGitHubUrlToolInput(BaseModel):
    """Input schema for the URL construction tool."""

    issue_number: str = Field(
        description="The GitHub issue number for 'google/adk-python'."
    )


class ConstructGitHubUrlToolOutput(BaseModel):
    """Output schema for the URL construction tool."""

    url: str
    error: Optional[str] = None


class ConstructGitHubUrlTool(BaseTool):
    """A tool that constructs the full GitHub URL for a given issue number."""

    def __init__(self):
        super().__init__(
            name="construct_github_url_tool",
            description="Constructs the GitHub URL for a given issue number.",
        )

    @override
    def _get_declaration(self):
        """Defines the tool's interface for the LLM."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ConstructGitHubUrlToolInput.model_json_schema(),
        )

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Executes the tool's URL construction logic."""
        tool_context.actions.skip_summarization = True
        try:
            input_data = ConstructGitHubUrlToolInput.model_validate(args)
            issue_number = input_data.issue_number
            # Hardcoded for the specific 'google/adk-python' repository this agent specializes in.
            url = f"<https://github.com/google/adk-python/issues/{issue_number}>"
            logger.info(f"Tool: {self.name} - Constructed URL: {url}")
            return ConstructGitHubUrlToolOutput(url=url).model_dump(exclude_none=True)
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error: {e}. Args: {args}", exc_info=True)
            return ConstructGitHubUrlToolOutput(
                url="", error=f"Error constructing URL: {e}"
            ).model_dump(exclude_none=True)
