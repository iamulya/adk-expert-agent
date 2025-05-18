import os
import asyncio
from typing import Any, Dict
from dotenv import load_dotenv

from google.cloud import secretmanager
from google.adk.tools import BaseTool, ToolContext
from google.adk.agents.llm_agent import LlmAgent as AdkLlmAgent # Alias to avoid conflict

# browser-use imports
from browser_use import Agent as BrowserUseAgent
from browser_use import Browser, BrowserConfig, BrowserContextConfig
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# --- Secret Manager Helper ---
_GEMINI_API_KEY = None

def get_gemini_api_key_from_secret_manager() -> str:
    """Fetches the Gemini API key from Google Cloud Secret Manager."""
    global _GEMINI_API_KEY
    if _GEMINI_API_KEY:
        return _GEMINI_API_KEY

    project_id = os.getenv("GCP_PROJECT_ID")
    secret_id = os.getenv("GEMINI_API_KEY_SECRET_ID")
    version_id = os.getenv("GEMINI_API_KEY_SECRET_VERSION", "1")

    if not project_id or not secret_id:
        raise ValueError(
            "GCP_PROJECT_ID and GEMINI_API_KEY_SECRET_ID must be set in .env"
        )

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    
    print(f"Fetching secret: {name}")
    try:
        response = client.access_secret_version(name=name)
        _GEMINI_API_KEY = response.payload.data.decode("UTF-8")
        os.environ["GOOGLE_API_KEY"] = _GEMINI_API_KEY # For langchain/browser-use
        print("Successfully fetched API key from Secret Manager.")
        return _GEMINI_API_KEY
    except Exception as e:
        print(f"Error fetching secret from Secret Manager: {e}")
        raise

# --- Browser-Use Tool ---
class ExtractGitHubIssueDetailsTool(BaseTool):
    """
    A tool to extract details (title, body, comments) from a GitHub issue URL
    using the browser-use library.
    """
    def __init__(self):
        super().__init__(
            name="extract_github_issue_details_tool",
            description="Extracts title, body, and all comments from a given GitHub issue URL."
        )
        # Ensure API key is loaded for the LLM used by browser-use
        get_gemini_api_key_from_secret_manager() 
        self.browser_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-0506", temperature=0)


    def _get_declaration(self):
        # This tool is called by the LLM, so it needs a declaration
        from google.genai import types as adk_types # ADK's genai types
        return adk_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=adk_types.Schema(
                type=adk_types.Type.OBJECT,
                properties={
                    "issue_url": adk_types.Schema(
                        type=adk_types.Type.STRING,
                        description="The full URL of the GitHub issue."
                    )
                },
                required=["issue_url"],
            ),
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, str]:
        issue_url = args.get("issue_url")
        if not issue_url:
            return {"error": "issue_url is required."}

        print(f"Tool: Extracting details from GitHub issue: {issue_url}")

        browser_task = (
            f"Go to the GitHub issue page: {issue_url}. "
            "Extract the issue title, the full body of the issue description, "
            "and the full text content of all comments. "
            "Present the extracted information clearly, labeling each part (Title, Body, Comments)."
        )
        
        # Configure browser-use to use a new browser context for each call
        # This ensures isolation if the tool is called multiple times.
        browser_config = BrowserConfig(
            headless=True, # Recommended for automation
            new_context_config=BrowserContextConfig(
                 # You might want to adjust timeouts if GitHub pages load slowly
                minimum_wait_page_load_time=2,
                maximum_wait_page_load_time=30
            )
        )
        
        # browser-use Agent
        # We need to manage the browser lifecycle within the tool call
        browser = Browser(config=browser_config)
        try:
            async with await browser.new_context() as bu_context:
                agent = BrowserUseAgent(
                    task=browser_task,
                    llm=self.browser_llm,
                    browser_context=bu_context, 
                    use_vision=True # Vision can be helpful for complex GitHub pages
                )
                history = await agent.run(max_steps=15) # Adjust max_steps as needed
        finally:
            await browser.close() # Ensure browser is closed

        if history and history.is_done() and history.is_successful():
            final_content = history.final_result()
            if final_content:
                print(f"Tool: Successfully extracted content from {issue_url}")
                return {"extracted_details": final_content}
            else:
                return {"error": f"browser-use agent finished but returned no content from {issue_url}."}
        else:
            error_message = "browser-use agent failed to extract details."
            if history and history.has_errors():
                error_message += f" Errors: {history.errors()}"
            print(f"Tool: Error extracting content from {issue_url}. {error_message}")
            return {"error": error_message}

