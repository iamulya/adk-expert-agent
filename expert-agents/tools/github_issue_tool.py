"""
This module defines the primary tool for fetching GitHub issue details via the REST API.
"""

import logging
import requests
from typing import Dict, Any, override

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from .github_utils import get_github_pat_from_secret_manager

logger = logging.getLogger(__name__)


class GetGithubIssueDescriptionTool(BaseTool):
    """
    A tool to fetch the description of a specific GitHub issue using the GitHub REST API.

    This tool is the primary method used by the agent to get issue details, as it is
    more reliable and efficient than web scraping. It requires a GitHub Personal
    Access Token (PAT) for authenticated requests to avoid rate limiting and access
    repository data.
    """

    def __init__(self):
        super().__init__(
            name="get_github_issue_description",
            description="Fetches the description (body) of a GitHub issue given the repository owner, repository name, and issue number.",
        )

    @override
    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        """Defines the tool's interface for the LLM, specifying its name, description, and parameters."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "owner": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        default="google",
                        description="The owner of the GitHub repository (e.g., 'google').",
                    ),
                    "repo": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        default="adk-python",
                        description="The name of the GitHub repository (e.g., 'adk-python').",
                    ),
                    "issue_number": genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description="The number of the GitHub issue.",
                    ),
                },
                required=["issue_number"],
            ),
        )

    async def run_async(
        self, *, args: Dict[str, Any], tool_context: ToolContext
    ) -> Dict[str, Any]:
        """
        Executes the tool's logic to fetch the issue description from the GitHub API.

        Args:
            args: A dictionary of arguments provided by the LLM, matching the tool's schema.
            tool_context: The context of the tool call.

        Returns:
            A dictionary containing either the issue's 'description' or an 'error' message.
        """
        # This ensures the tool's raw output is returned, rather than being summarized by the LLM.
        tool_context.actions.skip_summarization = True
        owner = args.get("owner", "google")
        repo = args.get("repo", "adk-python")
        issue_number = args.get("issue_number")

        if not owner:
            return {"error": "Missing required argument: owner."}
        if not repo:
            return {"error": "Missing required argument: repo."}
        if issue_number is None:
            return {"error": "Missing required argument: issue_number."}

        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        # Fetches the GitHub PAT securely.
        GITHUB_PERSONAL_ACCESS_TOKEN = get_github_pat_from_secret_manager()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "google-adk-tool",
            "Authorization": f"token {GITHUB_PERSONAL_ACCESS_TOKEN}",
        }

        try:
            # Make the GET request to the GitHub API.
            response = requests.get(api_url, headers=headers, timeout=10)
            # Raise an exception for non-2xx status codes.
            response.raise_for_status()

            issue_data = response.json()
            description = issue_data.get("body")

            # If the issue body is empty (None), return an empty string.
            if description is None:
                return {"description": ""}
            return {"description": str(description)}

        except requests.exceptions.HTTPError as e:
            error_message = f"HTTP error occurred: {e.response.status_code}"
            try:
                # Try to get a more specific error message from the API response body.
                error_details = e.response.json()
                error_message += f" - {error_details.get('message', e.response.text)}"
            except ValueError:
                error_message += f" - {e.response.text}"
            logger.error(
                f"Tool: {self.name} - GitHub API HTTPError: {error_message} for URL {api_url}",
                exc_info=True,
            )
            return {"error": error_message}
        except requests.exceptions.Timeout:
            logger.error(
                f"Tool: {self.name} - GitHub API Timeout for URL {api_url}",
                exc_info=True,
            )
            return {"error": f"Request timed out while fetching issue from {api_url}."}
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Tool: {self.name} - GitHub API RequestException: {e} for URL {api_url}",
                exc_info=True,
            )
            return {"error": f"A network request failed: {e}"}
        except ValueError:  # This would catch JSONDecodeError
            logger.error(
                f"Tool: {self.name} - GitHub API JSONDecodeError for URL {api_url}",
                exc_info=True,
            )
            return {"error": "Failed to decode JSON response from GitHub API."}
        except Exception as e:
            logger.error(
                f"Tool: {self.name} - GitHub API Unexpected error: {e} for URL {api_url}",
                exc_info=True,
            )
            return {"error": f"An unexpected error occurred: {str(e)}"}


# Create a singleton instance of the tool for easy import and use by agents.
get_github_issue_description = GetGithubIssueDescriptionTool()
