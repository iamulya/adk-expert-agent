"""
This module defines a tool for extracting GitHub issue content by scraping the webpage.

Note: This tool uses the `browser-use` library, which provides an agentic
wrapper around a headless browser (Playwright). It is an alternative method for
fetching issue details and is NOT the primary method used by the agent. The
primary, more reliable method is the `GetGithubIssueDescriptionTool` which uses
the GitHub REST API. This tool is kept as an example of browser-based interaction.
"""

import logging
from typing import Any, Dict, Optional, override
from pydantic import BaseModel

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

# browser-use provides an agent that can control a headless browser.
from browser_use import (
    Agent as BrowserUseAgent,
    Browser,
    BrowserConfig,
    BrowserContextConfig,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import DEFAULT_MODEL_NAME
from .github_utils import get_github_pat_from_secret_manager

logger = logging.getLogger(__name__)


class ExtractGitHubIssueDetailsToolInput(BaseModel):
    """Input schema for the web extraction tool."""

    url: str
    error: Optional[str] = None


class ExtractionResultInput(BaseModel):
    """
    Defines the output schema of this tool. This same model is used as an
    input schema for the `HandleExtractionResultTool` that would follow this
    in a tool chain.
    """

    extracted_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


class ExtractGitHubIssueDetailsTool(BaseTool):
    """A tool that uses a browser to fetch and extract content from a GitHub issue URL."""

    def __init__(self):
        super().__init__(
            name="fetch_webpage_content_for_github_issue_tool",
            description="Fetches and extracts relevant content (like title, body, comments) from a given GitHub issue URL.",
        )
        # The browser-use agent requires its own LLM instance to interpret the webpage.
        get_github_pat_from_secret_manager()
        self.browser_llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL_NAME, temperature=0
        )

    @override
    def _get_declaration(self):
        """Defines the tool's interface for the LLM."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ExtractGitHubIssueDetailsToolInput.model_json_schema(),
        )

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Executes the browser-based extraction logic."""
        tool_context.actions.skip_summarization = True
        try:
            input_data = ExtractGitHubIssueDetailsToolInput.model_validate(args)
            url = input_data.url
            previous_error = input_data.error
        except Exception as e:
            logger.error(
                f"Tool: {self.name} - Invalid input args: {args}. Error: {e}",
                exc_info=True,
            )
            return ExtractionResultInput(
                error=f"Invalid input to tool {self.name}: {e}"
            ).model_dump(exclude_none=True)

        if previous_error:
            logger.error(
                f"Tool: {self.name} - Received error from previous step: {previous_error}"
            )
            return ExtractionResultInput(error=previous_error).model_dump(
                exclude_none=True
            )

        if not url:
            logger.error(
                f"Tool: {self.name} - URL not provided in args and no previous error."
            )
            return ExtractionResultInput(
                error="URL is required for ExtractGitHubIssueDetailsTool."
            ).model_dump(exclude_none=True)

        logger.info(f"Tool: {self.name} - Extracting details from GitHub issue: {url}")
        # Define the task for the browser-use agent.
        browser_task = (
            f"Go to the GitHub issue page: {url}. "
            "Extract the full body of the issue description and all comments. Return ONLY that extracted text."
        )
        # Note: This requires Playwright browsers to be installed (`playwright install --with-deps chromium`).
        browser_config = BrowserConfig(
            headless=True,
            new_context_config=BrowserContextConfig(
                minimum_wait_page_load_time=2, maximum_wait_page_load_time=30
            ),
        )
        browser = Browser(config=browser_config)
        history = None
        try:
            async with await browser.new_context() as bu_context:
                # Create and run the browser-use agent.
                agent = BrowserUseAgent(
                    task=browser_task,
                    llm=self.browser_llm,
                    browser_context=bu_context,
                    use_vision=True,
                )
                history = await agent.run(max_steps=15)
        except Exception as e:
            logger.error(
                f"Tool: {self.name} - Exception during browser operation for {url}: {e}",
                exc_info=True,
            )
            return ExtractionResultInput(
                error=f"An error occurred while trying to browse {url}: {e}"
            ).model_dump(exclude_none=True)
        finally:
            await browser.close()

        # Process the result from the browser-use agent's run.
        if history and history.is_done() and history.is_successful():
            final_content = history.final_result()
            if final_content:
                logger.info(
                    f"Tool: {self.name} - Successfully extracted content from {url}"
                )
                return ExtractionResultInput(
                    extracted_details=final_content
                ).model_dump(exclude_none=True)
            else:
                logger.warning(
                    f"Tool: {self.name} - browser-use agent finished but returned no content from {url}."
                )
                return ExtractionResultInput(
                    message=f"Browser-use agent finished but returned no content from {url}. The page might be empty or inaccessible."
                ).model_dump(exclude_none=True)
        else:
            # Handle failure cases.
            error_message = "browser-use agent failed to extract details."
            if history and history.has_errors():
                error_details = ", ".join(map(str, history.errors()))
                error_message += f" Errors: {error_details}"
            elif history:
                error_message += f" Status: {history.status()}"
            else:
                error_message = f"browser-use agent did not run successfully for {url}. An earlier error might have occurred."
            logger.error(
                f"Tool: {self.name} - Error extracting content from {url}. {error_message}"
            )
            return ExtractionResultInput(error=error_message).model_dump(
                exclude_none=True
            )
