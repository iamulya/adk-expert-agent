#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- 0. Define base path ---
BASE_PATH="expert-agents"
AGENTS_PATH="$BASE_PATH/agents"
TOOLS_PATH="$BASE_PATH/tools"

echo "Starting refactoring of $BASE_PATH..."

# --- 1. Create new directories ---
echo "Creating directories: $AGENTS_PATH and $TOOLS_PATH"
mkdir -p "$AGENTS_PATH"
mkdir -p "$TOOLS_PATH"

# --- 2. Create __init__.py files for new directories ---
echo "Creating __init__.py files..."
touch "$AGENTS_PATH/__init__.py"
touch "$TOOLS_PATH/__init__.py"

# --- 3. Move existing agent files and rename ---
echo "Moving and renaming agent files..."
if [ -f "$BASE_PATH/document_generator_agent.py" ]; then
    mv "$BASE_PATH/document_generator_agent.py" "$AGENTS_PATH/document_generator_agent.py"
else
    echo "Warning: $BASE_PATH/document_generator_agent.py not found."
fi

if [ -f "$BASE_PATH/sequential_issue_processor.py" ]; then
    mv "$BASE_PATH/sequential_issue_processor.py" "$AGENTS_PATH/github_issue_processing_agent.py"
else
    echo "Warning: $BASE_PATH/sequential_issue_processor.py not found."
fi

# --- 4. Create new files for tools & populate them ---

# --- expert-agents/tools/github_utils.py ---
echo "Creating $TOOLS_PATH/github_utils.py"
cat <<EOF > "$TOOLS_PATH/github_utils.py"
# expert-agents/tools/github_utils.py
import os
from google.cloud import secretmanager
import logging

logger = logging.getLogger(__name__)

_GITHUB_PAT = None

def get_github_pat_from_secret_manager() -> str:
    global _GITHUB_PAT
    if _GITHUB_PAT:
        return _GITHUB_PAT
    project_number = os.getenv("GCP_PROJECT_NUMBER")
    secret_id = os.getenv("GITHUB_API_PAT_SECRET_ID")
    version_id = os.getenv("GITHUB_API_PAT_SECRET_VERSION", "1")
    if not project_number or not secret_id:
        raise ValueError("GCP_PROJECT_NUMBER and GITHUB_API_PAT_SECRET_ID must be set in .env")
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_number}/secrets/{secret_id}/versions/{version_id}"
    logger.info(f"Fetching GitHub PAT secret: {name}")
    try:
        response = client.access_secret_version(name=name)
        _GITHUB_PAT = response.payload.data.decode("UTF-8")
        logger.info("Successfully fetched GitHub PAT from Secret Manager.")
        return _GITHUB_PAT
    except Exception as e:
        logger.error(f"Error fetching GitHub PAT from Secret Manager: {e}", exc_info=True)
        raise

BOILERPLATE_STRINGS_TO_REMOVE = [
    "Is your feature request related to a problem? Please describe.",
    "Describe the solution you'd like",
    "Describe alternatives you've considered",
    "Describe the bug",
    "Minimal Reproduction",
    "Minimal steps to reproduce",
    "Desktop (please complete the following information):",
    "Please make sure you read the contribution guide and file the issues in the rigth place.",
    "To Reproduce",
    "Expected behavior",
    "Screenshots",
    "Additional context"
]
EOF

# --- expert-agents/tools/marp_utils.py ---
echo "Creating $TOOLS_PATH/marp_utils.py"
cat <<EOF > "$TOOLS_PATH/marp_utils.py"
# expert-agents/tools/marp_utils.py
import os
import asyncio
import logging
import subprocess
import tempfile
import uuid
import datetime
from typing import Optional

from google.cloud import storage
from google.adk.sessions.state import State
from google.adk.tools import ToolContext
from google.genai import types as genai_types

from ..config import (
    GCS_BUCKET_NAME, GCS_PROJECT_ID_FOR_BUCKET,
    GCS_SIGNED_URL_SA_EMAIL, SIGNED_URL_EXPIRATION_SECONDS
)

logger = logging.getLogger(__name__)

MARP_CLI_COMMAND = "marp"
GENERATED_DOCS_SUBDIR = "generated_documents_from_adk_agent"
DOC_LINK_STATE_KEY = State.TEMP_PREFIX + "generated_document_signed_url"

def _check_marp_cli():
    """Checks if marp-cli is installed and accessible."""
    try:
        subprocess.run([MARP_CLI_COMMAND, "--version"], check=True, capture_output=True)
        logger.info(f"'{MARP_CLI_COMMAND}' found and working.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(f"ERROR: '{MARP_CLI_COMMAND}' not found or not executable.")
        return False

async def _upload_to_gcs_and_get_signed_url(
    local_file_path: str,
    gcs_object_name_base: str,
    content_type: str,
    file_extension: str,
    tool_context: Optional[ToolContext] = None,
    markdown_content_for_fallback: Optional[str] = None
) -> str:
    session_id_for_path = tool_context._invocation_context.session.id if tool_context and hasattr(tool_context, '_invocation_context') else "unknown_session"
    unique_id = uuid.uuid4().hex[:8]
    
    gcs_object_name_base_cleaned = gcs_object_name_base.strip('/').replace(file_extension, '')
    gcs_object_name = f"adk_generated_documents/{session_id_for_path}/{gcs_object_name_base_cleaned}_{unique_id}{file_extension}"
    adk_artifact_filename = f"{os.path.basename(gcs_object_name_base_cleaned)}_{unique_id}{file_extension}"

    if GCS_BUCKET_NAME and GCS_PROJECT_ID_FOR_BUCKET:
        try:
            logger.info(f"Attempting to save to GCS bucket '{GCS_BUCKET_NAME}' as object '{gcs_object_name}'")
            storage_client = storage.Client(project=GCS_PROJECT_ID_FOR_BUCKET)
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(gcs_object_name)
            
            blob.upload_from_filename(local_file_path, content_type=content_type)
            logger.info(f"File successfully uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}")

            signed_url_str = "No signed URL generated if SA email is not set."
            if GCS_SIGNED_URL_SA_EMAIL:
                from google import auth
                principal_credentials, _ = auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                from google.auth import impersonated_credentials
                impersonated_target_credentials = impersonated_credentials.Credentials(
                        source_credentials=principal_credentials,
                        target_principal=GCS_SIGNED_URL_SA_EMAIL,
                        target_scopes=['https://www.googleapis.com/auth/devstorage.read_write'],
                        lifetime=SIGNED_URL_EXPIRATION_SECONDS + 60
                    )
                
                signed_url_str = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS),
                    method="GET",
                    credentials=impersonated_target_credentials,
                )
                logger.info(f"Generated GCS signed URL: {signed_url_str}")
                output = f"Document generated and saved to Google Cloud Storage. Download (link expires in {SIGNED_URL_EXPIRATION_SECONDS // 60} mins): {signed_url_str}"
                if tool_context:
                    tool_context.state[DOC_LINK_STATE_KEY] = output
                    logger.info(f"Saved signed URL to tool context state under key: {DOC_LINK_STATE_KEY}")
                    tool_context.actions.skip_summarization = True
                return output
            else:
                logger.warning("GCS_SIGNED_URL_SA_EMAIL not configured. Returning public GCS path instead of signed URL.")
                public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_name}"
                return f"Document generated and saved to Google Cloud Storage at: gs://{GCS_BUCKET_NAME}/{gcs_object_name} (Public URL if bucket/object is public: {public_url}). Signed URL generation skipped as service account for signing is not configured."

        except Exception as e:
            logger.error(f"Error during GCS storage or signed URL generation for {gcs_object_name}: {e}", exc_info=True)
            if tool_context and markdown_content_for_fallback:
                logger.warning("Falling back to ADK in-memory artifact service.")
                await tool_context.save_artifact(adk_artifact_filename + ".md", genai_types.Part(text=markdown_content_for_fallback))
                return f"Error saving to GCS. Source Markdown saved to ADK artifacts as: {adk_artifact_filename}.md. GCS Error: {str(e)}"
            return f"Error saving to GCS: {str(e)}. Fallback to ADK artifact not possible without tool_context/markdown."
    else:
        if tool_context and markdown_content_for_fallback:
            logger.warning("GCS bucket/project not configured. Saving source Markdown to ADK in-memory artifact service.")
            await tool_context.save_artifact(adk_artifact_filename + ".md", genai_types.Part(text=markdown_content_for_fallback))
            return f"GCS not configured. Source Markdown for {adk_artifact_filename} saved as ADK artifact: {adk_artifact_filename}.md."
        logger.warning("GCS bucket/project not configured. Cannot save artifact as tool_context or markdown_content is missing.")
        return "GCS not configured and could not save to ADK artifacts."
EOF

# --- expert-agents/tools/marp_document_tools.py ---
echo "Creating $TOOLS_PATH/marp_document_tools.py"
cat <<EOF > "$TOOLS_PATH/marp_document_tools.py"
# expert-agents/tools/marp_document_tools.py
import os
import asyncio
import logging
import subprocess
import tempfile
from typing import Optional

from google.adk.tools import ToolContext
from .marp_utils import (
    _check_marp_cli, _upload_to_gcs_and_get_signed_url,
    MARP_CLI_COMMAND, GENERATED_DOCS_SUBDIR, DOC_LINK_STATE_KEY
)

logger = logging.getLogger(__name__)

async def _generate_with_marp_and_upload(
    markdown_content: str,
    output_filename_base: str, # Base name without extension
    marp_format_flag: str, # --pdf, --html, --pptx
    gcs_content_type: str,
    file_extension: str, # .pdf, .html, .pptx
    tool_context: Optional[ToolContext] = None
) -> str:
    if not _check_marp_cli():
        return "Error: marp-cli is not installed or accessible. Cannot generate document."

    local_marp_output_dir = os.path.join(tempfile.gettempdir(), GENERATED_DOCS_SUBDIR)
    os.makedirs(local_marp_output_dir, exist_ok=True)
    
    safe_local_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in output_filename_base)
    local_output_path = os.path.join(local_marp_output_dir, safe_local_filename + file_extension)
    
    tmp_md_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".md", encoding='utf-8') as tmp_md_file:
            tmp_md_file.write(markdown_content)
            tmp_md_file_path = tmp_md_file.name
        
        cmd = [MARP_CLI_COMMAND, tmp_md_file_path, marp_format_flag, "-o", local_output_path, "--allow-local-files"]
        logger.info(f"Executing Marp command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        
        if os.path.exists(local_output_path):
            logger.info(f"Marp generated file locally: {local_output_path}. Marp stdout: {result.stdout}")
            return await _upload_to_gcs_and_get_signed_url(
                local_file_path=local_output_path,
                gcs_object_name_base=output_filename_base,
                content_type=gcs_content_type,
                file_extension=file_extension,
                tool_context=tool_context,
                markdown_content_for_fallback=markdown_content
            )
        else:
            logger.error(f"Marp file generation commanded, but output file '{local_output_path}' not found. Marp stdout: {result.stdout}. Stderr: {result.stderr}")
            return f"Marp file generation commanded, but output file not found. Marp output: {result.stdout}. Error: {result.stderr}"

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running marp-cli for {marp_format_flag}: {e.stderr}", exc_info=True)
        return f"Error generating document with marp-cli: {e.stderr}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during {marp_format_flag} generation: {str(e)}", exc_info=True)
        return f"An unexpected error occurred during document generation: {str(e)}"
    finally:
        if tmp_md_file_path and os.path.exists(tmp_md_file_path):
            os.remove(tmp_md_file_path)
        if 'local_output_path' in locals() and os.path.exists(local_output_path):
             try:
                os.remove(local_output_path)
             except Exception as e_rem:
                 logger.warning(f"Could not remove temporary local Marp file {local_output_path}: {e_rem}")

async def generate_pdf_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--pdf", "application/pdf", ".pdf", tool_context
    )

async def generate_html_slides_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--html", "text/html", ".html", tool_context
    )

async def generate_pptx_slides_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx", tool_context
    )
EOF

# --- expert-agents/tools/github_issue_tool.py ---
echo "Creating $TOOLS_PATH/github_issue_tool.py"
cat <<EOF > "$TOOLS_PATH/github_issue_tool.py"
# expert-agents/tools/github_issue_tool.py
import logging
import requests
from typing import Dict, Any, override

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from .github_utils import get_github_pat_from_secret_manager

logger = logging.getLogger(__name__)

class GetGithubIssueDescriptionTool(BaseTool):
  """
  A tool to fetch the description of a specific GitHub issue.
  """
  def __init__(self):
    super().__init__(
        name="get_github_issue_description",
        description="Fetches the description (body) of a GitHub issue given the repository owner, repository name, and issue number.",
    )

  @override
  def _get_declaration(self) -> genai_types.FunctionDeclaration:
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
    GITHUB_PERSONAL_ACCESS_TOKEN = get_github_pat_from_secret_manager()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "google-adk-tool",
        "Authorization": f"token {GITHUB_PERSONAL_ACCESS_TOKEN}",
    }

    try:
      response = requests.get(api_url, headers=headers, timeout=10)
      response.raise_for_status()
      
      issue_data = response.json()
      description = issue_data.get("body")

      if description is None:
        return {"description": ""}
      return {"description": str(description)}
      
    except requests.exceptions.HTTPError as e:
      error_message = f"HTTP error occurred: {e.response.status_code}"
      try:
          error_details = e.response.json()
          error_message += f" - {error_details.get('message', e.response.text)}"
      except ValueError:
          error_message += f" - {e.response.text}"
      logger.error(f"Tool: {self.name} - GitHub API HTTPError: {error_message} for URL {api_url}", exc_info=True)
      return {"error": error_message}
    except requests.exceptions.Timeout:
        logger.error(f"Tool: {self.name} - GitHub API Timeout for URL {api_url}", exc_info=True)
        return {"error": f"Request timed out while fetching issue from {api_url}."}
    except requests.exceptions.RequestException as e:
        logger.error(f"Tool: {self.name} - GitHub API RequestException: {e} for URL {api_url}", exc_info=True)
        return {"error": f"A network request failed: {e}"}
    except ValueError:
        logger.error(f"Tool: {self.name} - GitHub API JSONDecodeError for URL {api_url}", exc_info=True)
        return {"error": "Failed to decode JSON response from GitHub API."}
    except Exception as e:
        logger.error(f"Tool: {self.name} - GitHub API Unexpected error: {e} for URL {api_url}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

get_github_issue_description = GetGithubIssueDescriptionTool()
EOF

# --- expert-agents/tools/adk_guidance_tool.py ---
echo "Creating $TOOLS_PATH/adk_guidance_tool.py"
cat <<EOF > "$TOOLS_PATH/adk_guidance_tool.py"
# expert-agents/tools/adk_guidance_tool.py
import logging
from typing import Any, Dict, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import DEFAULT_MODEL_NAME, API_KEY
from ..context_loader import get_escaped_adk_context_for_llm

logger = logging.getLogger(__name__)

class ADKGuidanceToolInput(BaseModel):
    document_text: str = Field(description="The cleaned GitHub issue text or other document for ADK guidance.")

class ADKGuidanceToolOutput(BaseModel):
    guidance: str

class ADKGuidanceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="adk_guidance_tool",
            description="Provides ADK guidance based on the provided document text and ADK context."
        )
        self.llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL_NAME, temperature=0.1, google_api_key=API_KEY)
        self.adk_context_for_llm = get_escaped_adk_context_for_llm()

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ADKGuidanceToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        tool_context.actions.skip_summarization = True
        try:
            input_data = ADKGuidanceToolInput.model_validate(args)
            document_text = input_data.document_text

            if document_text.startswith("Error fetching issue details:") or \
               document_text.startswith("Message from issue fetcher:") or \
               document_text.startswith("Error: No content extracted") or \
               document_text == "GitHub issue content was empty after cleaning." or \
               document_text.startswith("Error retrieving GitHub issue:"):
                logger.warning(f"Tool: {self.name} received prior error/message: {document_text}")
                return ADKGuidanceToolOutput(guidance=document_text).model_dump()

            if not document_text.strip():
                guidance_message = "The provided document text was empty. Cannot provide ADK guidance."
                logger.warning(f"Tool: {self.name} - {guidance_message}")
                return ADKGuidanceToolOutput(guidance=guidance_message).model_dump()

            prompt = f\"\"\"
You are an expert on Google's Agent Development Kit (ADK) version 1.0.0.
Your task is to provide feedback based on your comprehensive ADK knowledge and the provided document text.

Your ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{self.adk_context_for_llm}
--- END OF ADK CONTEXT ---

Provided Document Text:
--- START OF DOCUMENT TEXT ---
{document_text}
--- END OF DOCUMENT TEXT ---

Analyze the "Provided Document Text" in conjunction with "Your ADK Knowledge Context".
Formulate a helpful, detailed, and actionable response based SOLELY on these two pieces of information.
This is your final response to be presented to the user. Do not ask further questions or try to use tools.
\"\"\"
            logger.info(f"Tool: {self.name} - Prompting LLM with document (first 100 chars): {document_text[:100]}")
            response = await self.llm.ainvoke(prompt)
            guidance = response.content
            logger.info(f"Tool: {self.name} - Guidance received (first 100 chars): {guidance[:100]}")
            return ADKGuidanceToolOutput(guidance=guidance).model_dump()
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error generating guidance: {e}. Args: {args}", exc_info=True)
            return ADKGuidanceToolOutput(guidance=f"Error generating ADK guidance: {e}").model_dump()
EOF

# --- expert-agents/tools/construct_github_url_tool.py ---
echo "Creating $TOOLS_PATH/construct_github_url_tool.py"
cat <<EOF > "$TOOLS_PATH/construct_github_url_tool.py"
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
EOF

# --- expert-agents/tools/extract_github_issue_details_tool.py ---
echo "Creating $TOOLS_PATH/extract_github_issue_details_tool.py"
cat <<EOF > "$TOOLS_PATH/extract_github_issue_details_tool.py"
# expert-agents/tools/extract_github_issue_details_tool.py
import logging
from typing import Any, Dict, Optional, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from browser_use import Agent as BrowserUseAgent, Browser, BrowserConfig, BrowserContextConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import DEFAULT_MODEL_NAME
from .github_utils import get_github_pat_from_secret_manager # Assuming PAT might be needed by browser_use for private repos or rate limits

logger = logging.getLogger(__name__)

class ExtractGitHubIssueDetailsToolInput(BaseModel):
    url: str
    error: Optional[str] = None

class ExtractionResultInput(BaseModel): # This model is used as output by this tool
    extracted_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

class ExtractGitHubIssueDetailsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="fetch_webpage_content_for_github_issue_tool",
            description="Fetches and extracts relevant content (like title, body, comments) from a given GitHub issue URL."
        )
        # Ensure API key for browser_use's LLM is available if it's different from ADK's default setup
        # from ..config import API_KEY # if needed explicitly
        get_github_pat_from_secret_manager() # Call to ensure PAT is loaded if browser-use needs it implicitly or for future use
        self.browser_llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL_NAME, temperature=0) # Uses API_KEY from config

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ExtractGitHubIssueDetailsToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        tool_context.actions.skip_summarization = True
        try:
            input_data = ExtractGitHubIssueDetailsToolInput.model_validate(args)
            url = input_data.url
            previous_error = input_data.error
        except Exception as e:
            logger.error(f"Tool: {self.name} - Invalid input args: {args}. Error: {e}", exc_info=True)
            return ExtractionResultInput(error=f"Invalid input to tool {self.name}: {e}").model_dump(exclude_none=True)

        if previous_error:
            logger.error(f"Tool: {self.name} - Received error from previous step: {previous_error}")
            return ExtractionResultInput(error=previous_error).model_dump(exclude_none=True)
        
        if not url:
            logger.error(f"Tool: {self.name} - URL not provided in args and no previous error.")
            return ExtractionResultInput(error="URL is required for ExtractGitHubIssueDetailsTool.").model_dump(exclude_none=True)
        
        logger.info(f"Tool: {self.name} - Extracting details from GitHub issue: {url}")
        browser_task = (
            f"Go to the GitHub issue page: {url}. "
            "Extract the full body of the issue description and all comments. Return ONLY that extracted text."
        )
        # Note: browser-use might require Playwright browsers to be installed.
        # playwright install --with-deps chromium
        browser_config = BrowserConfig(
            headless=True,
            new_context_config=BrowserContextConfig(
                minimum_wait_page_load_time=2,
                maximum_wait_page_load_time=30
            )
        )
        browser = Browser(config=browser_config)
        history = None
        try:
            async with await browser.new_context() as bu_context:
                agent = BrowserUseAgent(
                    task=browser_task,
                    llm=self.browser_llm,
                    browser_context=bu_context,
                    use_vision=True
                )
                history = await agent.run(max_steps=15)
        except Exception as e:
            logger.error(f"Tool: {self.name} - Exception during browser operation for {url}: {e}", exc_info=True)
            return ExtractionResultInput(error=f"An error occurred while trying to browse {url}: {e}").model_dump(exclude_none=True)
        finally:
            await browser.close()
            
        if history and history.is_done() and history.is_successful():
            final_content = history.final_result()
            if final_content:
                logger.info(f"Tool: {self.name} - Successfully extracted content from {url}")
                return ExtractionResultInput(extracted_details=final_content).model_dump(exclude_none=True)
            else:
                logger.warning(f"Tool: {self.name} - browser-use agent finished but returned no content from {url}.")
                return ExtractionResultInput(message=f"Browser-use agent finished but returned no content from {url}. The page might be empty or inaccessible.").model_dump(exclude_none=True)
        else:
            error_message = "browser-use agent failed to extract details."
            if history and history.has_errors():
                error_details = ", ".join(map(str, history.errors()))
                error_message += f" Errors: {error_details}"
            elif history:
                error_message += f" Status: {history.status()}"
            else:
                error_message = f"browser-use agent did not run successfully for {url}. An earlier error might have occurred."
            logger.error(f"Tool: {self.name} - Error extracting content from {url}. {error_message}")
            return ExtractionResultInput(error=error_message).model_dump(exclude_none=True)
EOF

# --- expert-agents/tools/handle_extraction_result_tool.py ---
echo "Creating $TOOLS_PATH/handle_extraction_result_tool.py"
cat <<EOF > "$TOOLS_PATH/handle_extraction_result_tool.py"
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
EOF

# --- expert-agents/tools/clean_github_issue_text_tool.py ---
echo "Creating $TOOLS_PATH/clean_github_issue_text_tool.py"
cat <<EOF > "$TOOLS_PATH/clean_github_issue_text_tool.py"
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
        if text.startswith("Error fetching issue details:") or \
           text.startswith("Message from issue fetcher:") or \
           text.startswith("Error: No content extracted"):
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
EOF

# --- expert-agents/tools/mermaid_to_png_and_upload_tool.py ---
echo "Creating $TOOLS_PATH/mermaid_to_png_and_upload_tool.py"
cat <<EOF > "$TOOLS_PATH/mermaid_to_png_and_upload_tool.py"
# expert-agents/tools/mermaid_to_png_and_upload_tool.py
import asyncio
import datetime
import logging
import os
import re
import subprocess
import tempfile
import uuid
from typing import Dict

from google.cloud import storage
import google.auth.exceptions # No longer need google.oauth2.service_account or google.auth.transport.requests

from google.adk.sessions.state import State
from google.adk.tools import BaseTool, ToolContext
import google.genai.types as genai_types

from ..config import ( # Relative import
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID_FOR_BUCKET,
    GCS_SIGNED_URL_SA_EMAIL,
    SIGNED_URL_EXPIRATION_SECONDS,
    MERMAID_CLI_PATH
)

logger = logging.getLogger(__name__)
GCS_LINK_STATE_KEY = State.TEMP_PREFIX + "gcs_link_for_diagram"

PUPPETEER_CONFIG_PATH = os.getenv("PUPPETEER_CONFIG_PATH", "/app/puppeteer-config.json")

class MermaidToPngAndUploadTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="mermaid_to_png_and_gcs_upload",
            description="Converts Mermaid diagram syntax to a PNG image, uploads it to Google Cloud Storage, and returns a signed URL for access.",
        )

    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "mermaid_syntax": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The Mermaid diagram syntax to convert, enclosed in \`\`\`mermaid ... \`\`\`.",
                    )
                },
                required=["mermaid_syntax"],
            ),
        )

    async def run_async(self, args: Dict[str, str], tool_context: ToolContext) -> str:
        mermaid_syntax_raw = args.get("mermaid_syntax")
        if not mermaid_syntax_raw:
            return "Error: Mermaid syntax is required."

        match = re.search(r"\`\`\`mermaid\s*([\s\S]+?)\s*\`\`\`", mermaid_syntax_raw, re.DOTALL)
        if match:
            mermaid_syntax = match.group(1).strip()
        else:
            mermaid_syntax = mermaid_syntax_raw.strip()

        if not mermaid_syntax:
            return "Error: Extracted Mermaid syntax is empty."

        session_id = tool_context._invocation_context.session.id if tool_context and hasattr(tool_context, '_invocation_context') and hasattr(tool_context._invocation_context, 'session') else "unknown_session"
        
        png_data = None
        mmd_file_path = None
        png_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".mmd", encoding='utf-8') as mmd_file:
                mmd_file.write(mermaid_syntax)
                mmd_file_path = mmd_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as png_temp_obj:
                png_file_path = png_temp_obj.name

            logger.info(f"Running mmdc: {MERMAID_CLI_PATH} -p {PUPPETEER_CONFIG_PATH} -i {mmd_file_path} -o {png_file_path}")
            process = await asyncio.create_subprocess_exec(
                MERMAID_CLI_PATH,
                "-p", PUPPETEER_CONFIG_PATH,
                "-i", mmd_file_path,
                "-o", png_file_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = f"Mermaid CLI error (code {process.returncode}): {stderr.decode() if stderr else stdout.decode()}"
                logger.error(error_msg)
                if tool_context:
                    await tool_context.save_artifact(
                        f"error_mermaid_syntax_{uuid.uuid4().hex[:8]}.mmd",
                        genai_types.Part(text=mermaid_syntax)
                    )
                return f"Error converting Mermaid to PNG: {error_msg}. The original syntax was saved as an artifact."

            with open(png_file_path, "rb") as f:
                png_data = f.read()
            logger.info(f"Successfully converted Mermaid to PNG: {png_file_path}")

        except FileNotFoundError:
            logger.error(f"Mermaid CLI (mmdc) not found at '{MERMAID_CLI_PATH}'. Please install it or set MERMAID_CLI_PATH.")
            return f"Error: Mermaid CLI (mmdc) not found at '{MERMAID_CLI_PATH}'. Cannot generate PNG."
        except Exception as e:
            logger.error(f"Error during PNG generation: {e}", exc_info=True)
            return f"Error generating PNG: {str(e)}"
        finally:
            if mmd_file_path and os.path.exists(mmd_file_path):
                os.remove(mmd_file_path)
            if png_file_path and os.path.exists(png_file_path):
                os.remove(png_file_path)

        if not png_data:
            return "Error: PNG data could not be generated."

        gcs_object_name = f"mermaid_diagrams/{session_id}/diagram_{uuid.uuid4().hex[:8]}.png"
        adk_artifact_filename = f"mermaid_diagram_{uuid.uuid4().hex[:8]}.png"

        if GCS_BUCKET_NAME and GCS_PROJECT_ID_FOR_BUCKET:
            try:
                logger.info(f"Attempting to save PNG to GCS bucket '{GCS_BUCKET_NAME}' in project '{GCS_PROJECT_ID_FOR_BUCKET}' as object '{gcs_object_name}'")
                storage_client = storage.Client(project=GCS_PROJECT_ID_FOR_BUCKET)
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(gcs_object_name)
                blob.upload_from_string(png_data, content_type='image/png')
                logger.info(f"PNG successfully uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}")

                if not GCS_SIGNED_URL_SA_EMAIL:
                    logger.warning("GCS_SIGNED_URL_SA_EMAIL not set. Cannot generate signed URL.")
                    public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_name}"
                    return f"Diagram PNG uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}. Public URL (if public): {public_url}. Signed URL not generated as SA email is missing."

                try:
                    from google import auth
                    principal_credentials, _ = auth.default(
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    from google.auth import impersonated_credentials
                    impersonated_target_credentials = impersonated_credentials.Credentials(
                            source_credentials=principal_credentials,
                            target_principal=GCS_SIGNED_URL_SA_EMAIL,
                            target_scopes=['https://www.googleapis.com/auth/devstorage.read_write'],
                            lifetime=120
                        )
                    signed_url_str = blob.generate_signed_url(
                        version="v4",
                        expiration=datetime.timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS),
                        method="GET",
                        credentials=impersonated_target_credentials,
                    )
                    logger.info(f"Generated signed URL: {signed_url_str}")
                    output = f"Diagram PNG generated and saved to Google Cloud Storage. Download (link expires in {SIGNED_URL_EXPIRATION_SECONDS // 60} mins): {signed_url_str}"
                    if tool_context:
                        tool_context.state[GCS_LINK_STATE_KEY] = output
                        logger.info(f"Saved signed URL to tool context state under key: {GCS_LINK_STATE_KEY}")
                        tool_context.actions.skip_summarization = True
                    return output
                except google.auth.exceptions.RefreshError as refresh_err:
                    logger.error(f"Error refreshing impersonated credentials for Mermaid: {refresh_err}. Check SA '{GCS_SIGNED_URL_SA_EMAIL}'.", exc_info=True)
                    if tool_context:
                        await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                    return f"Error generating signed URL for Mermaid diagram. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(refresh_err)}"
                except Exception as e_sign:
                    logger.error(f"Error during signed URL generation for Mermaid: {e_sign}", exc_info=True)
                    if tool_context:
                        await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                    return f"Error generating signed URL for Mermaid diagram. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(e_sign)}"
            except Exception as e_gcs:
                logger.error(f"Error during GCS PNG storage for Mermaid: {e_gcs}", exc_info=True)
                if tool_context:
                    await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                return f"Error saving Mermaid diagram to GCS. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(e_gcs)}"
        else:
            logger.info("GCS bucket/project for Mermaid not configured. Saving PNG to ADK in-memory artifact service.")
            if tool_context:
                await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
            return f"Mermaid diagram PNG generated and saved as ADK artifact: {adk_artifact_filename}."

mermaid_gcs_tool_instance = MermaidToPngAndUploadTool()
EOF

# --- expert-agents/tools/prepare_document_content_tool.py ---
echo "Creating $TOOLS_PATH/prepare_document_content_tool.py"
cat <<EOF > "$TOOLS_PATH/prepare_document_content_tool.py"
# expert-agents/tools/prepare_document_content_tool.py
import logging
from typing import Any, Dict, Literal, override
from pydantic import BaseModel, Field, ValidationError

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

class PrepareDocumentContentToolInput(BaseModel):
    markdown_content: str = Field(description="The generated Markdown content for the document.")
    document_type: Literal["pdf", "html", "pptx"] = Field(description="The type of document requested by the user (pdf, html, or pptx).")
    output_filename_base: str = Field(description="A base name for the output file, e.g., 'adk_report'. The document_generator_agent will append the correct extension.")
    original_user_query: str = Field(description="The original user query that requested the document.")

class PrepareDocumentContentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="prepare_document_content_tool",
            description="Gathers generated markdown content, document type, and filename base. This tool is called by the orchestrator agent; its output triggers the actual document generation agent."
        )

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=PrepareDocumentContentToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        logger.info(f"PrepareDocumentContentTool 'run_async' called with args: {str(args)[:200]}. Returning these args directly.")
        try:
            validated_args = PrepareDocumentContentToolInput.model_validate(args)
            return validated_args.model_dump()
        except ValidationError as ve:
            logger.error(f"PrepareDocumentContentTool: LLM provided invalid arguments: {ve}. Args: {args}", exc_info=True)
            return {"error": f"Invalid arguments from LLM for content preparation: {str(ve)}", "original_args": args}
EOF

# --- expert-agents/tools/__init__.py (content) ---
echo "Populating $TOOLS_PATH/__init__.py"
cat <<EOF > "$TOOLS_PATH/__init__.py"
# expert-agents/tools/__init__.py
from .adk_guidance_tool import ADKGuidanceTool, ADKGuidanceToolInput, ADKGuidanceToolOutput
from .clean_github_issue_text_tool import CleanGitHubIssueTextTool, CleanGitHubIssueTextToolInput, CleanGitHubIssueTextToolOutput
from .construct_github_url_tool import ConstructGitHubUrlTool, ConstructGitHubUrlToolInput, ConstructGitHubUrlToolOutput
from .extract_github_issue_details_tool import ExtractGitHubIssueDetailsTool, ExtractGitHubIssueDetailsToolInput, ExtractionResultInput
from .github_issue_tool import GetGithubIssueDescriptionTool, get_github_issue_description
from .github_utils import get_github_pat_from_secret_manager, BOILERPLATE_STRINGS_TO_REMOVE
from .handle_extraction_result_tool import HandleExtractionResultTool, HandleExtractionResultToolOutput
from .marp_document_tools import (
    generate_pdf_from_markdown_with_gcs,
    generate_html_slides_from_markdown_with_gcs,
    generate_pptx_slides_from_markdown_with_gcs,
)
from .marp_utils import DOC_LINK_STATE_KEY
from .mermaid_to_png_and_upload_tool import MermaidToPngAndUploadTool, GCS_LINK_STATE_KEY, mermaid_gcs_tool_instance
from .prepare_document_content_tool import PrepareDocumentContentTool, PrepareDocumentContentToolInput

__all__ = [
    "ADKGuidanceTool", "ADKGuidanceToolInput", "ADKGuidanceToolOutput",
    "CleanGitHubIssueTextTool", "CleanGitHubIssueTextToolInput", "CleanGitHubIssueTextToolOutput",
    "ConstructGitHubUrlTool", "ConstructGitHubUrlToolInput", "ConstructGitHubUrlToolOutput",
    "ExtractGitHubIssueDetailsTool", "ExtractGitHubIssueDetailsToolInput", "ExtractionResultInput",
    "GetGithubIssueDescriptionTool", "get_github_issue_description",
    "get_github_pat_from_secret_manager", "BOILERPLATE_STRINGS_TO_REMOVE",
    "HandleExtractionResultTool", "HandleExtractionResultToolOutput",
    "generate_pdf_from_markdown_with_gcs",
    "generate_html_slides_from_markdown_with_gcs",
    "generate_pptx_slides_from_markdown_with_gcs",
    "DOC_LINK_STATE_KEY",
    "MermaidToPngAndUploadTool", "GCS_LINK_STATE_KEY", "mermaid_gcs_tool_instance",
    "PrepareDocumentContentTool", "PrepareDocumentContentToolInput",
]
EOF

# --- 5. Create new files for agents & populate them ---

# --- expert-agents/agents/mermaid_syntax_verifier_agent.py ---
echo "Creating $AGENTS_PATH/mermaid_syntax_verifier_agent.py"
cat <<EOF > "$AGENTS_PATH/mermaid_syntax_verifier_agent.py"
# expert-agents/agents/mermaid_syntax_verifier_agent.py
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from ..config import DEFAULT_MODEL_NAME # Relative import

class MermaidSyntaxVerifierAgentToolInput(BaseModel):
    mermaid_syntax: str = Field(description="The Mermaid syntax to verify and correct.")

mermaid_syntax_verifier_agent = ADKAgent(
    name="mermaid_syntax_verifier_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=(
        "You are an expert in Mermaid diagram syntax. "
        "Your sole responsibility is to receive Mermaid syntax, verify its correctness, "
        "and correct any syntax errors. "
        "If the provided syntax is already correct, return it as is. "
        "If there are errors, return the corrected Mermaid syntax. "
        "Your output MUST ONLY be the Mermaid code block itself (e.g., \`\`\`mermaid\\n...\\n\`\`\`). "
        "Do not add any other explanations, greetings, or conversational text."
    ),
    description="Verifies and corrects Mermaid diagram syntax. Expects input as a JSON string with a 'mermaid_syntax' key.",
    input_schema=MermaidSyntaxVerifierAgentToolInput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)
EOF

# --- expert-agents/agents/diagram_generator_agent.py ---
echo "Creating $AGENTS_PATH/diagram_generator_agent.py"
cat <<EOF > "$AGENTS_PATH/diagram_generator_agent.py"
# expert-agents/agents/diagram_generator_agent.py
import logging
import json
from pydantic import BaseModel, Field, ValidationError

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from ..context_loader import get_escaped_adk_context_for_llm # Relative
from ..config import PRO_MODEL_NAME # Relative
from ..callbacks import log_prompt_before_model_call # Relative
from .mermaid_syntax_verifier_agent import mermaid_syntax_verifier_agent # Relative
from ..tools.mermaid_to_png_and_upload_tool import mermaid_gcs_tool_instance, GCS_LINK_STATE_KEY # Relative

logger = logging.getLogger(__name__)

class DiagramGeneratorAgentToolInput(BaseModel):
    diagram_query: str = Field(description="The user's query describing the architecture diagram to be generated.")

def diagram_generator_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_diagram_query = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            try:
                input_data = DiagramGeneratorAgentToolInput.model_validate_json(invocation_ctx.user_content.parts[0].text)
                user_diagram_query = input_data.diagram_query
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(f"DiagramGeneratorAgent: Error parsing input: {e}. Raw input: {invocation_ctx.user_content.parts[0].text[:100]}")
                return "Error: Could not understand the diagram request. Please provide a clear query for the diagram."
    
    if not user_diagram_query:
        return "Error: No diagram query provided. Please specify what kind of diagram you need."

    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    instruction = f\"\"\"
You are an AI assistant that generates architecture diagrams in Mermaid syntax. You have access to the ADK (Agent Development Kit) knowledge context.

ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---

The user's request for the diagram is: "{user_diagram_query}"

Follow these steps precisely:
1.  Based on the user's request ("{user_diagram_query}") and the ADK Knowledge Context, generate the Mermaid syntax for the diagram.
    The Mermaid syntax MUST be enclosed in a standard Mermaid code block, like so:
    \`\`\`mermaid
    graph TD
        A --> B
    \`\`\`
2.  After generating the initial Mermaid syntax, you MUST call the '{mermaid_syntax_verifier_agent.name}' tool.
    The input to this tool MUST be a JSON object with a single key "mermaid_syntax", where the value is the Mermaid syntax you just generated (including the \`\`\`mermaid ... \`\`\` block).
    Example call: {{"mermaid_syntax": "\`\`\`mermaid\\ngraph TD; A-->B;\\n\`\`\`"}}
3.  The '{mermaid_syntax_verifier_agent.name}' tool will return the verified (and potentially corrected) Mermaid syntax.
    This response will also be a JSON object, and you need to extract the value of the "mermaid_syntax" key from its "result" field.
4.  Once you have the final, verified Mermaid syntax, you MUST call the '{mermaid_gcs_tool_instance.name}' tool.
    The input to this tool MUST be a JSON object with a single key "mermaid_syntax", where the value is the final verified Mermaid syntax (including the \`\`\`mermaid ... \`\`\` block).
5.  Your final response for this turn MUST ONLY be the result from the '{mermaid_gcs_tool_instance.name}' tool (which will be a GCS signed URL or an error message).
    Do not add any conversational fluff, explanations, or greetings around this final URL.
\"\"\"
    return instruction

async def diagram_generator_after_agent_cb(callback_context: CallbackContext) -> genai_types.Content | None:
    gcs_link = callback_context.state.get(GCS_LINK_STATE_KEY)
    if gcs_link:
        logger.info(f"DiagramGeneratorAgent (after_agent_callback): Returning GCS link: {gcs_link}")
        return genai_types.Content(parts=[genai_types.Part(text=str(gcs_link))])
    logger.warning("DiagramGeneratorAgent (after_agent_callback): GCS link not found in state.")
    return genai_types.Content(parts=[genai_types.Part(text="Error: Could not generate diagram link.")])

diagram_generator_agent = ADKAgent(
    name="mermaid_diagram_orchestrator_agent",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=diagram_generator_agent_instruction_provider,
    tools=[
        AgentTool(agent=mermaid_syntax_verifier_agent),
        mermaid_gcs_tool_instance,
    ],
    input_schema=DiagramGeneratorAgentToolInput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=diagram_generator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)
EOF

# --- expert-agents/agents/__init__.py (content) ---
echo "Populating $AGENTS_PATH/__init__.py"
cat <<EOF > "$AGENTS_PATH/__init__.py"
# expert-agents/agents/__init__.py
from .diagram_generator_agent import diagram_generator_agent, DiagramGeneratorAgentToolInput
from .document_generator_agent import document_generator_agent, DocumentGeneratorAgentToolInput
from .github_issue_processing_agent import github_issue_processing_agent, GitHubIssueProcessingInput, SequentialProcessorFinalOutput
from .mermaid_syntax_verifier_agent import mermaid_syntax_verifier_agent, MermaidSyntaxVerifierAgentToolInput

__all__ = [
    "diagram_generator_agent", "DiagramGeneratorAgentToolInput",
    "document_generator_agent", "DocumentGeneratorAgentToolInput",
    "github_issue_processing_agent", "GitHubIssueProcessingInput", "SequentialProcessorFinalOutput",
    "mermaid_syntax_verifier_agent", "MermaidSyntaxVerifierAgentToolInput",
]
EOF

# --- 6. Update expert-agents/agent.py (root_agent module) ---
echo "Updating $BASE_PATH/agent.py"
cat <<EOF > "$BASE_PATH/agent.py"
# expert-agents/agent.py
import os
import logging
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Literal, Optional, override

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

from .context_loader import get_escaped_adk_context_for_llm
from .config import DEFAULT_MODEL_NAME, PRO_MODEL_NAME
from .callbacks import log_prompt_before_model_call

# Import agents from the new 'agents' subpackage
from .agents.github_issue_processing_agent import github_issue_processing_agent, GitHubIssueProcessingInput, SequentialProcessorFinalOutput
from .agents.document_generator_agent import document_generator_agent, DocumentGeneratorAgentToolInput
from .agents.diagram_generator_agent import diagram_generator_agent, DiagramGeneratorAgentToolInput

# Import tools from the new 'tools' subpackage
from .tools.prepare_document_content_tool import PrepareDocumentContentTool, PrepareDocumentContentToolInput
# Note: mermaid_gcs_tool_instance and GCS_LINK_STATE_KEY are used by diagram_generator_agent,
# which is now in its own file and imports them directly. So, no need to import them here for root_agent.

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# Pydantic model for PrepareDocumentContentTool's output when used by root_agent's LLM
# This is the same as PrepareDocumentContentToolInput, but named for clarity in root_agent context
class PreparedContentDataForDocGen(PrepareDocumentContentToolInput):
    pass

async def root_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any
) -> Optional[Any]:

    if tool.name == github_issue_processing_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_processing_agent.name}'.")
        tool_context.actions.skip_summarization = True
        response_text = "Error: Could not process response from sequential agent."
        try:
            if isinstance(tool_response, str):
                 response_dict = json.loads(tool_response)
            elif isinstance(tool_response, dict):
                 response_dict = tool_response
            else:
                raise ValueError(f"Unexpected tool_response type: {type(tool_response)}")

            validated_output = SequentialProcessorFinalOutput.model_validate(response_dict)
            response_text = validated_output.guidance
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(f"RootAgent: Error parsing/validating response from sequential agent: {e}. Response: {str(tool_response)[:200]}", exc_info=True)
            if isinstance(tool_response, str) and ("error" in tool_response.lower() or not tool_response.strip().startswith("{")):
                response_text = tool_response
            else:
                 response_text = f"Error: Sequential agent returned an unexpected structure: {str(tool_response)[:200]}"
        logger.info(f"RootAgent (after_tool_callback): Relaying guidance from sequential agent: {response_text[:200]}...")
        return genai_types.Content(parts=[genai_types.Part(text=response_text)])

    elif tool.name == document_generator_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Received response from '{document_generator_agent.name}': {str(tool_response)[:200]}")
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Document generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    elif tool.name == diagram_generator_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Received response from '{diagram_generator_agent.name}': {str(tool_response)[:200]}")
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Diagram generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    elif tool.name == "prepare_document_content_tool":
        logger.info(f"RootAgent (after_tool_callback): 'prepare_document_content_tool' completed. Output from tool: {str(tool_response)[:200]}")
        tool_context.actions.skip_summarization = False
        return tool_response

    logger.warning(f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}")
    return None


def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_query_text = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
    
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events:
        last_event = invocation_ctx.session.events[-1]
        if last_event.author == root_agent.name and \
           last_event.content and last_event.content.parts and \
           last_event.content.parts[0].function_response and \
           last_event.content.parts[0].function_response.name == "prepare_document_content_tool":
            
            logger.info("RootAgent (instruction_provider): Detected response from prepare_document_content_tool. Instructing to call document_generator_agent.")
            tool_output_data_container = last_event.content.parts[0].function_response.response
            
            prepared_content_data = tool_output_data_container.get("result", tool_output_data_container)

            try:
                # Validate against the Pydantic model for what prepare_document_content_tool returns
                validated_content_for_doc_gen = PreparedContentDataForDocGen.model_validate(prepared_content_data)
                # Map to DocumentGeneratorAgentToolInput
                doc_gen_agent_actual_input = DocumentGeneratorAgentToolInput(
                    markdown_content=validated_content_for_doc_gen.markdown_content,
                    document_type=validated_content_for_doc_gen.document_type,
                    output_filename=validated_content_for_doc_gen.output_filename_base # document_generator_agent expects 'output_filename'
                )
                system_instruction = f\"\"\"
You have received structured data from the 'prepare_document_content_tool'.
The data is: {json.dumps(prepared_content_data)}
Your task is to call the tool named '{document_generator_agent.name}'.
This tool expects arguments conforming to this schema:
{DocumentGeneratorAgentToolInput.model_json_schema()}
Based on the data received from 'prepare_document_content_tool', you MUST call the '{document_generator_agent.name}' tool with the correctly mapped arguments:
{doc_gen_agent_actual_input.model_dump_json()}
Your response should ONLY be the function call. Do not include any other text.
\"\"\"
                return system_instruction
            except (ValidationError, Exception) as e:
                logger.error(f"RootAgent (instruction_provider): Error processing data from prepare_document_content_tool for doc gen: {e}. Data: {str(prepared_content_data)[:200]}", exc_info=True)
                return "Error: Could not process the data from the content preparation step. Please try rephrasing your request."

    patterns = [
        re.compile(r"(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)(?:\s*(?:on|for|in|related to)\s*google/adk-python)?", re.IGNORECASE),
        re.compile(r"google/adk-python\s*(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(?:on|for|in|related to)\s*google/adk-python", re.IGNORECASE)
    ]
    extracted_issue_number = None
    for pattern in patterns:
        match = pattern.search(user_query_text)
        if match:
            for group_val in match.groups():
                if group_val and group_val.isdigit():
                    extracted_issue_number = group_val
                    break
        if extracted_issue_number:
            break
    
    is_github_keywords_present = "github" in user_query_text.lower() or \
                                 any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature"])

    doc_gen_keywords_pdf = ["pdf", "document", "report"]
    doc_gen_keywords_slides = ["slides", "presentation", "deck", "pptx", "powerpoint", "html slides"]
    requested_doc_type = None
    if any(kw in user_query_text.lower() for kw in doc_gen_keywords_pdf):
        requested_doc_type = "pdf"
    elif any(kw in user_query_text.lower() for kw in doc_gen_keywords_slides):
        requested_doc_type = "pptx" if "pptx" in user_query_text.lower() or "powerpoint" in user_query_text.lower() else "html"

    diagram_keywords = ["diagram", "architecture", "visualize", "mermaid"]
    is_diagram_request = any(kw in user_query_text.lower() for kw in diagram_keywords)

    if is_diagram_request:
        logger.info(f"RootAgent (instruction_provider): Detected architecture diagram request: '{user_query_text}'")
        diagram_agent_input_payload = DiagramGeneratorAgentToolInput(diagram_query=user_query_text).model_dump_json()
        system_instruction = f\"\"\"
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking for an architecture diagram. Their query is: "{user_query_text}"
Your task is to call the '{diagram_generator_agent.name}' tool.
The tool expects its input as a JSON string. The value for the "request" argument MUST be the following JSON string:
{diagram_agent_input_payload}
This is your only action for this turn. Output only the tool call.
\"\"\"
    elif requested_doc_type:
        logger.info(f"RootAgent (instruction_provider): Detected document generation request for type '{requested_doc_type}'. Query: '{user_query_text}'")
        system_instruction = f\"\"\"
You are an expert on Google's Agent Development Kit (ADK) version 1.0.0 and a document content creator.
You have access to a tool called '{PrepareDocumentContentTool().name}'.
ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
The user wants you to generate a document of type '{requested_doc_type}'. Their request is: "{user_query_text}"
Your tasks are:
1.  Analyze the user's request: "{user_query_text}".
2.  Using the ADK Knowledge Context, generate comprehensive Markdown content for \`marp-cli\`.
3.  Determine a suitable base filename (e.g., "adk_overview").
4.  You MUST call '{PrepareDocumentContentTool().name}' with "markdown_content", "document_type" ("{requested_doc_type}"), "output_filename_base", and "original_user_query".
Your response should ONLY be the function call.
\"\"\"
    elif extracted_issue_number:
        logger.info(f"RootAgent (instruction_provider): Found issue number '{extracted_issue_number}'. Instructing to call GitHubIssueProcessingSequentialAgent.")
        sequential_agent_input_payload_str = GitHubIssueProcessingInput(issue_number=extracted_issue_number).model_dump_json()
        system_instruction = f\"\"\"
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking about GitHub issue number {extracted_issue_number} for 'google/adk-python'.
Your task is to call the '{github_issue_processing_agent.name}' tool.
The tool expects a single argument named "request".
The value for the "request" argument MUST be the following JSON string:
{sequential_agent_input_payload_str}
Output only the tool call.
\"\"\"
    elif is_github_keywords_present:
        logger.info("RootAgent (instruction_provider): GitHub keywords present, but no issue number. Asking.")
        system_instruction = "Your final response for this turn MUST be exactly: 'It looks like you're asking about a GitHub issue for google/adk-python, but I couldn't find a specific issue number. Please provide the GitHub issue number.'"
    else:
        logger.info(f"RootAgent (instruction_provider): General ADK query: '{user_query_text}'")
        system_instruction = f\"\"\"
You are an expert on Google's Agent Development Kit (ADK) version 1.0.0.
Your primary role is to answer general questions about ADK.
When a user starts a conversation, greet them by introducing yourself as an ADK 1.0.0 expert.
Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general ADK questions):
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
\"\"\"
    return system_instruction

root_agent_tools = [
    AgentTool(agent=github_issue_processing_agent),
    AgentTool(agent=document_generator_agent),
    AgentTool(agent=diagram_generator_agent),
    PrepareDocumentContentTool(), # This tool is defined in .tools.prepare_document_content_tool
]

root_agent = ADKAgent(
    name="adk_expert_orchestrator",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=root_agent_instruction_provider,
    tools=root_agent_tools,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=root_agent_after_tool_callback,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=60000, # Consider if this is still appropriate
        top_p=0.6,
    )
)
EOF

# --- 7. Update moved agent files with new relative imports ---
echo "Updating imports in moved agent files..."

# --- expert-agents/agents/document_generator_agent.py (update imports) ---
if [ -f "$AGENTS_PATH/document_generator_agent.py" ]; then
    # Temporary file for sed
    TMP_AGENT_FILE="$AGENTS_PATH/document_generator_agent.py.tmp"

    # Replace 'from .tools import' with 'from ..tools.marp_document_tools import' and specific tool names
    # Replace 'from .config import' with 'from ..config import'
    # Replace 'from .callbacks import' with 'from ..callbacks import'
    sed \
        -e 's/from \.tools import DOC_LINK_STATE_KEY, generate_pdf_from_markdown_with_gcs, generate_html_slides_from_markdown_with_gcs, generate_pptx_slides_from_markdown_with_gcs/from ..tools.marp_document_tools import DOC_LINK_STATE_KEY, generate_pdf_from_markdown_with_gcs, generate_html_slides_from_markdown_with_gcs, generate_pptx_slides_from_markdown_with_gcs/' \
        -e 's/from \.config import/from ..config import/' \
        -e 's/from \.callbacks import/from ..callbacks import/' \
        "$AGENTS_PATH/document_generator_agent.py" > "$TMP_AGENT_FILE" && mv "$TMP_AGENT_FILE" "$AGENTS_PATH/document_generator_agent.py"
    echo "Updated imports in $AGENTS_PATH/document_generator_agent.py"
else
    echo "Warning: $AGENTS_PATH/document_generator_agent.py not found for import update."
fi

# --- expert-agents/agents/github_issue_processing_agent.py (update imports) ---
if [ -f "$AGENTS_PATH/github_issue_processing_agent.py" ]; then
    TMP_AGENT_FILE="$AGENTS_PATH/github_issue_processing_agent.py.tmp"
    sed \
        -e 's/from \.tools import GetGithubIssueDescriptionTool, ADKGuidanceTool, ADKGuidanceToolInput, ADKGuidanceToolOutput/from ..tools import GetGithubIssueDescriptionTool, ADKGuidanceTool, ADKGuidanceToolInput, ADKGuidanceToolOutput/' \
        -e 's/from \.config import/from ..config import/' \
        -e 's/from \.callbacks import/from ..callbacks import/' \
        "$AGENTS_PATH/github_issue_processing_agent.py" > "$TMP_AGENT_FILE" && mv "$TMP_AGENT_FILE" "$AGENTS_PATH/github_issue_processing_agent.py"
    echo "Updated imports in $AGENTS_PATH/github_issue_processing_agent.py"
else
    echo "Warning: $AGENTS_PATH/github_issue_processing_agent.py not found for import update."
fi


# --- 8. Update expert-agents/__init__.py ---
echo "Updating $BASE_PATH/__init__.py"
cat <<EOF > "$BASE_PATH/__init__.py"
# expert-agents/__init__.py
from .agent import root_agent
from .agents.github_issue_processing_agent import github_issue_processing_agent

__all__ = ["root_agent", "github_issue_processing_agent"]
EOF

# --- 9. Update README.md ---
echo "Updating README.md"
cat <<EOF > "README.md"
# My ADK Expert Agent

This agent is an expert on Google's Agent Development Kit (ADK) version 1.0.0.
It can answer general questions about ADK and help find solutions/guidance for GitHub issues.

## Directory Structure (Post-Refactor)

\`\`\`
└── iamulya-adk-expert-agent/
    ├── README.md
    ├── pyproject.toml
    ├── expert-agents/
    │   ├── __init__.py
    │   ├── agent.py  # Contains root_agent
    │   ├── callbacks.py
    │   ├── config.py
    │   ├── context_loader.py
    │   ├── Dockerfile
    │   ├── puppeteer-config.json
    │   ├── test.md
    │   ├── test.mmd
    │   ├── .env.example
    │   ├── data/
    │   │   └── google-adk-python-1.0.txt
    │   ├── agents/  # New folder for specific agents
    │   │   ├── __init__.py
    │   │   ├── diagram_generator_agent.py
    │   │   ├── document_generator_agent.py
    │   │   ├── github_issue_processing_agent.py
    │   │   └── mermaid_syntax_verifier_agent.py
    │   └── tools/   # New folder for tools
    │       ├── __init__.py
    │       ├── adk_guidance_tool.py
    │       ├── clean_github_issue_text_tool.py
    │       ├── construct_github_url_tool.py
    │       ├── extract_github_issue_details_tool.py
    │       ├── github_issue_tool.py
    │       ├── github_utils.py
    │       ├── handle_extraction_result_tool.py
    │       ├── marp_document_tools.py
    │       ├── marp_utils.py
    │       ├── mermaid_to_png_and_upload_tool.py
    │       └── prepare_document_content_tool.py
    ├── webui/
    │   └── ... (rest of webui structure remains the same)
    └── .github/
        └── ... (rest of .github structure remains the same)
\`\`\`

## Setup

1.  **Create a Python Virtual Environment**:
    It's recommended to use a virtual environment.
    \`\`\`bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    \`\`\`

2.  **Install Dependencies**:
    Make sure you have \`uv\` or \`pip\` installed.
    Using \`uv\` (recommended):
    \`\`\`bash
    pip install uv
    uv pip install -r requirements.txt  # Or directly from pyproject.toml if using uv sync
    # If pyproject.toml is set up for uv sync:
    # uv sync
    \`\`\`
    Using \`pip\`:
    \`\`\`bash
    pip install -e . 
    # This will install dependencies from pyproject.toml in editable mode.
    # Alternatively, generate a requirements.txt from pyproject.toml if needed.
    \`\`\`

3.  **Install Playwright Browsers (Optional for general use, if not using other browser-dependent tools)**:
    If you plan to use tools that rely on \`browser-use\` (e.g., the \`ExtractGitHubIssueDetailsTool\`, though the primary GitHub issue processing in this agent no longer uses it by default), you'll need Playwright browsers. Install Chromium (recommended for compatibility):
    \`\`\`bash
    playwright install --with-deps chromium
    \`\`\`

4.  **Configure Environment Variables**:
    *   Copy \`expert-agents/.env.example\` to \`expert-agents/.env\`.
    *   Fill in your Google Cloud Project ID (\`GCP_PROJECT_ID\`).
    *   Fill in the Secret ID (\`GEMINI_API_KEY_SECRET_ID\`) for your Gemini API key stored in Google Cloud Secret Manager.
    *   Fill in the Secret ID (\`GITHUB_API_PAT_SECRET_ID\`) for your GitHub Personal Access Token (PAT) stored in Google Cloud Secret Manager. This PAT needs read access to public repositories (e.g., for fetching issue details).
    *   Ensure the service account or user running the agent has permissions to access these secrets.

5.  **Google Cloud Authentication**:
    Ensure you are authenticated with Google Cloud, e.g., by running:
    \`\`\`bash
    gcloud auth application-default login
    \`\`\`

## Running the Agent

You can run this agent as a web application using the ADK CLI:

\`\`\`bash
adk web .
\`\`\`
EOF

# --- 10. Delete old tool files ---
echo "Deleting old tool files..."
if [ -f "$BASE_PATH/tools.py" ]; then
    rm "$BASE_PATH/tools.py"
    echo "Deleted $BASE_PATH/tools.py"
else
    echo "Warning: $BASE_PATH/tools.py not found for deletion."
fi

if [ -f "$BASE_PATH/mermaid_tool.py" ]; then
    rm "$BASE_PATH/mermaid_tool.py"
    echo "Deleted $BASE_PATH/mermaid_tool.py"
else
    echo "Warning: $BASE_PATH/mermaid_tool.py not found for deletion."
fi


echo "Refactoring complete!"
echo "Please review the changes, especially the import paths in Python files."
echo "You might need to adjust your Dockerfile if it specifically copied tools.py or mermaid_tool.py."
echo "Consider running a linter/formatter like pylint/black/ruff."