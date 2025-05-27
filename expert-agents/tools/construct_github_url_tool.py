# expert-agents/tools/construct_github_url_tool.py
import logging
from typing import Any, Dict, Optional, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

class ConstructGitHubUrlToolInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

class ConstructGitHubUrlToolOutput(BaseModel):
    url: str
    error: Optional[str] = None

class ConstructGitHubUrlTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="construct_github_url_tool",
            description="Constructs the GitHub URL for a given issue number."
        )

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ConstructGitHubUrlToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        tool_context.actions.skip_summarization = True
        try:
            input_data = ConstructGitHubUrlToolInput.model_validate(args)
            issue_number = input_data.issue_number
            url = f"https://github.com/google/adk-python/issues/{issue_number}"
            logger.info(f"Tool: {self.name} - Constructed URL: {url}")
            return ConstructGitHubUrlToolOutput(url=url).model_dump(exclude_none=True)
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error: {e}. Args: {args}", exc_info=True)
            return ConstructGitHubUrlToolOutput(url="", error=f"Error constructing URL: {e}").model_dump(exclude_none=True)
