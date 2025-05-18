import os
import asyncio
from typing import Any, Dict
from dotenv import load_dotenv

from google.cloud import secretmanager
from google.adk.tools import BaseTool, ToolContext
from browser_use import Agent as BrowserUseAgent
from browser_use import Browser, BrowserConfig, BrowserContextConfig
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_GEMINI_API_KEY = None

def get_gemini_api_key_from_secret_manager() -> str:
    global _GEMINI_API_KEY
    if _GEMINI_API_KEY:
        return _GEMINI_API_KEY
    project_id = os.getenv("GCP_PROJECT_ID")
    secret_id = os.getenv("GEMINI_API_KEY_SECRET_ID")
    version_id = os.getenv("GEMINI_API_KEY_SECRET_VERSION", "latest")
    if not project_id or not secret_id:
        raise ValueError("GCP_PROJECT_ID and GEMINI_API_KEY_SECRET_ID must be set in .env")
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/1"
    print(f"Fetching secret: {name}")
    try:
        response = client.access_secret_version(name=name)
        _GEMINI_API_KEY = response.payload.data.decode("UTF-8")
        os.environ["GOOGLE_API_KEY"] = _GEMINI_API_KEY
        print("Successfully fetched API key from Secret Manager.")
        return _GEMINI_API_KEY
    except Exception as e:
        print(f"Error fetching secret from Secret Manager: {e}")
        raise

class ExtractGitHubIssueDetailsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="fetch_webpage_content_for_github_issue",
            description="Fetches and extracts relevant content (like title, body, comments) from a given GitHub issue URL."
        )
        get_gemini_api_key_from_secret_manager() 
        self.browser_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06", temperature=0)

    def _get_declaration(self):
        from google.genai import types as adk_types
        return adk_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=adk_types.Schema(
                type=adk_types.Type.OBJECT,
                properties={
                    "url": adk_types.Schema(
                        type=adk_types.Type.STRING,
                        description="The full URL of the GitHub issue page."
                    ),
                },
                required=["url"],
            ),
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, str]:
        url = args.get("url")
        if not url:
            return {"error": "url is required."}
        print(f"Tool: Extracting details from GitHub issue: {url}")
        browser_task = (
            f"Go to the GitHub issue page: {url}. "
            "Extract the full body of the issue description and return ONLY that."
        )
        browser_config = BrowserConfig(
            headless=True,
            new_context_config=BrowserContextConfig(
                minimum_wait_page_load_time=2,
                maximum_wait_page_load_time=30
            )
        )
        browser = Browser(config=browser_config)
        try:
            async with await browser.new_context() as bu_context:
                agent = BrowserUseAgent(
                    task=browser_task,
                    llm=self.browser_llm,
                    browser_context=bu_context, 
                    use_vision=True
                )
                history = await agent.run(max_steps=15)
        finally:
            await browser.close()
        if history and history.is_done() and history.is_successful():
            final_content = history.final_result()
            if final_content:
                print(f"Tool: Successfully extracted content from {url}")
                return {"extracted_details": final_content}
            else:
                return {"error": f"browser-use agent finished but returned no content from {url}."}
        else:
            error_message = "browser-use agent failed to extract details."
            if history and history.has_errors():
                error_message += f" Errors: {history.errors()}"
            print(f"Tool: Error extracting content from {url}. {error_message}")
            return {"error": error_message}