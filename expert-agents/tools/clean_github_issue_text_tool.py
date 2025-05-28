# expert-agents/tools/clean_github_issue_text_tool.py
import logging
import re
from typing import Any, Dict, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from .github_utils import BOILERPLATE_STRINGS_TO_REMOVE

logger = logging.getLogger(__name__)

class CleanGitHubIssueTextToolInput(BaseModel):
    raw_text: str = Field(description="The raw text content of the GitHub issue.")

class CleanGitHubIssueTextToolOutput(BaseModel):
    cleaned_text: str

class CleanGitHubIssueTextTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="clean_github_issue_text_tool",
            description="Cleans boilerplate and extra newlines from GitHub issue text."
        )

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=CleanGitHubIssueTextToolInput.model_json_schema()
        )
    
    def clean_text_logic(self, text: str) -> str:
        if not text: return ""
        if text.startswith("Error fetching issue details:") or            text.startswith("Message from issue fetcher:") or            text.startswith("Error: No content extracted"):
            return text
            
        cleaned_text = text
        for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
            cleaned_text = cleaned_text.replace(boilerplate, "")
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
        if not cleaned_text:
            return "GitHub issue content was empty after cleaning."
        return cleaned_text

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        tool_context.actions.skip_summarization = True
        try:
            input_data = CleanGitHubIssueTextToolInput.model_validate(args)
            cleaned_text = self.clean_text_logic(input_data.raw_text)
            logger.info(f"Tool: {self.name} - Cleaned text (first 100 chars): {cleaned_text[:100]}")
            return CleanGitHubIssueTextToolOutput(cleaned_text=cleaned_text).model_dump()
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error cleaning text: {e}. Args: {args}", exc_info=True)
            return CleanGitHubIssueTextToolOutput(cleaned_text=f"Error cleaning text: {e}").model_dump()
