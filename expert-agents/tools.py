import os
import asyncio
import re
import logging
from typing import Any, Dict, Optional, override
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import subprocess
import tempfile
import shutil
import uuid
import datetime

from google.cloud import secretmanager, storage
from google.adk.sessions.state import State 
from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types # For FunctionDeclaration
from browser_use import Agent as BrowserUseAgent
from browser_use import Browser, BrowserConfig, BrowserContextConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import (
    DEFAULT_MODEL_NAME, API_KEY,
    GCS_BUCKET_NAME, GCS_PROJECT_ID_FOR_BUCKET,
    GCS_SIGNED_URL_SA_EMAIL, SIGNED_URL_EXPIRATION_SECONDS
)
from .context_loader import get_escaped_adk_context_for_llm

load_dotenv()
logger = logging.getLogger(__name__)

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

_GITHUB_PAT = None 

MARP_CLI_COMMAND = "marp"
GENERATED_DOCS_SUBDIR = "generated_documents_from_adk_agent" # For local fallback

DOC_LINK_STATE_KEY = State.TEMP_PREFIX + "generated_document_signed_url" # Key to store signed URL in tool context state

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
    print(f"Fetching GitHub PAT secret: {name}") 
    try:
        response = client.access_secret_version(name=name)
        _GITHUB_PAT = response.payload.data.decode("UTF-8") 
        print("Successfully fetched GitHub PAT from Secret Manager.") 
        return _GITHUB_PAT
    except Exception as e:
        print(f"Error fetching GitHub PAT from Secret Manager: {e}") 
        raise

# --- Helper function to check for marp-cli ---
def _check_marp_cli():
    """Checks if marp-cli is installed and accessible."""
    try:
        subprocess.run([MARP_CLI_COMMAND, "--version"], check=True, capture_output=True)
        logger.info(f"'{MARP_CLI_COMMAND}' found and working.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(f"ERROR: '{MARP_CLI_COMMAND}' not found or not executable.")
        return False

# --- Common GCS Upload and Signed URL Logic ---
async def _upload_to_gcs_and_get_signed_url(
    local_file_path: str, 
    gcs_object_name_base: str, # e.g., "adk_docs/session_xyz/my_report" (no extension)
    content_type: str,
    file_extension: str, # .pdf, .html, .pptx
    tool_context: Optional[ToolContext] = None, # For ADK artifact fallback
    markdown_content_for_fallback: Optional[str] = None
) -> str:
    """
    Uploads a file to GCS and returns a signed URL. Falls back to ADK artifacts if GCS fails.
    """
    session_id_for_path = tool_context._invocation_context.session.id if tool_context and hasattr(tool_context, '_invocation_context') else "unknown_session"
    unique_id = uuid.uuid4().hex[:8]
    
    # Ensure gcs_object_name_base does not have leading/trailing slashes from filename
    gcs_object_name_base_cleaned = gcs_object_name_base.strip('/').replace(file_extension, '')

    gcs_object_name = f"adk_generated_documents/{session_id_for_path}/{gcs_object_name_base_cleaned}_{unique_id}{file_extension}"
    
    # For ADK artifact fallback, use a simpler name structure
    adk_artifact_filename = f"{os.path.basename(gcs_object_name_base_cleaned)}_{unique_id}{file_extension}"


    if GCS_BUCKET_NAME and GCS_PROJECT_ID_FOR_BUCKET:
        try:
            logger.info(f"Attempting to save to GCS bucket '{GCS_BUCKET_NAME}' as object '{gcs_object_name}'")
            storage_client = storage.Client(project=GCS_PROJECT_ID_FOR_BUCKET)
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(gcs_object_name)
            
            blob.upload_from_filename(local_file_path, content_type=content_type)
            logger.info(f"File successfully uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}")

            signed_url = "No signed URL generated if SA email is not set."
            if GCS_SIGNED_URL_SA_EMAIL:
                from google import auth
                # Use ADC of the environment running the agent to impersonate
                principal_credentials, _ = auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )

                from google.auth import impersonated_credentials
                impersonated_target_credentials = impersonated_credentials.Credentials(
                        source_credentials=principal_credentials,
                        target_principal=GCS_SIGNED_URL_SA_EMAIL,
                        target_scopes=['https://www.googleapis.com/auth/devstorage.read_write'],
                        lifetime=SIGNED_URL_EXPIRATION_SECONDS + 60 # Add a bit of buffer
                    )
                
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS),
                    method="GET",
                    credentials=impersonated_target_credentials,
                )
                logger.info(f"Generated GCS signed URL: {signed_url}")
                output = f"Document generated and saved to Google Cloud Storage. Download (link expires in {SIGNED_URL_EXPIRATION_SECONDS // 60} mins): {signed_url}"
                # Save the signed URL in the tool context state for later retrieval
                if tool_context:
                    tool_context.state[DOC_LINK_STATE_KEY] = output
                    logger.info(f"Saved signed URL to tool context state under key: {DOC_LINK_STATE_KEY}")

                    # This tool's output itself doesn't need summarization BY DiagramGeneratorAgent's LLM
                    tool_context.actions.skip_summarization = True
                return output
            else:
                logger.warning("GCS_SIGNED_URL_SA_EMAIL not configured. Returning public GCS path instead of signed URL.")
                public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_name}"
                # Note: This URL will only work if the object is publicly readable or the user has direct GCS access.
                return f"Document generated and saved to Google Cloud Storage at: gs://{GCS_BUCKET_NAME}/{gcs_object_name} (Public URL if bucket/object is public: {public_url}). Signed URL generation skipped as service account for signing is not configured."

        except Exception as e:
            logger.error(f"Error during GCS storage or signed URL generation for {gcs_object_name}: {e}", exc_info=True)
            if tool_context and markdown_content_for_fallback: # Fallback to ADK artifact
                logger.warning("Falling back to ADK in-memory artifact service.")
                # Save the source markdown as artifact for debugging, not the binary file directly.
                # If you want to save the binary, read local_file_path and create Part from bytes.
                await tool_context.save_artifact(adk_artifact_filename + ".md", genai_types.Part(text=markdown_content_for_fallback))
                return f"Error saving to GCS. Source Markdown saved to ADK artifacts as: {adk_artifact_filename}.md. GCS Error: {str(e)}"
            return f"Error saving to GCS: {str(e)}. Fallback to ADK artifact not possible without tool_context/markdown."
    else: # GCS not configured, use ADK artifact service as fallback
        if tool_context and markdown_content_for_fallback:
            logger.warning("GCS bucket/project not configured. Saving source Markdown to ADK in-memory artifact service.")
            await tool_context.save_artifact(adk_artifact_filename + ".md", genai_types.Part(text=markdown_content_for_fallback))
            return f"GCS not configured. Source Markdown for {adk_artifact_filename} saved as ADK artifact: {adk_artifact_filename}.md."
        logger.warning("GCS bucket/project not configured. Cannot save artifact as tool_context or markdown_content is missing.")
        return "GCS not configured and could not save to ADK artifacts."

# --- Marp + GCS Tool Functions ---
async def _generate_with_marp_and_upload(
    markdown_content: str, 
    output_filename_base: str, # Base name without extension
    marp_format_flag: str, # --pdf, --html, --pptx
    gcs_content_type: str,
    file_extension: str, # .pdf, .html, .pptx
    tool_context: Optional[ToolContext] = None # Passed for ADK artifact fallback
) -> str:
    """Generic function to run Marp and upload to GCS."""
    if not _check_marp_cli():
        return "Error: marp-cli is not installed or accessible. Cannot generate document."

    # Ensure local output directory exists for marp-cli
    local_marp_output_dir = os.path.join(tempfile.gettempdir(), GENERATED_DOCS_SUBDIR)
    os.makedirs(local_marp_output_dir, exist_ok=True)
    
    # Sanitize filename and ensure correct extension for local file
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
            # Now upload and get signed URL
            return await _upload_to_gcs_and_get_signed_url(
                local_file_path=local_output_path,
                gcs_object_name_base=output_filename_base, # Pass base name for GCS path
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
        if 'local_output_path' in locals() and os.path.exists(local_output_path): # Clean up local Marp output
             try:
                os.remove(local_output_path)
             except Exception as e_rem:
                 logger.warning(f"Could not remove temporary local Marp file {local_output_path}: {e_rem}")


async def generate_pdf_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    """
    Generates a PDF from Markdown, uploads to GCS, and returns a signed URL.
    The output_filename should be the base name, e.g., 'my_report'.
    """
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--pdf", "application/pdf", ".pdf", tool_context
    )

async def generate_html_slides_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    """
    Generates HTML slides from Markdown, uploads to GCS, and returns a signed URL.
    The output_filename should be the base name, e.g., 'my_presentation'.
    """
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--html", "text/html", ".html", tool_context
    )

async def generate_pptx_slides_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    """
    Generates PPTX slides from Markdown, uploads to GCS, and returns a signed URL.
    The output_filename should be the base name, e.g., 'my_deck'.
    """
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx", tool_context
    )

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

    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ConstructGitHubUrlToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]: 
        tool_context.actions.skip_summarization = True # Ensure flow termination
        try:
            input_data = ConstructGitHubUrlToolInput.model_validate(args)
            issue_number = input_data.issue_number
            url = f"https://github.com/google/adk-python/issues/{issue_number}"
            logger.info(f"Tool: {self.name} - Constructed URL: {url}")
            return ConstructGitHubUrlToolOutput(url=url).model_dump(exclude_none=True)
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error: {e}. Args: {args}", exc_info=True)
            return ConstructGitHubUrlToolOutput(url="", error=f"Error constructing URL: {e}").model_dump(exclude_none=True)


class ExtractionResultInput(BaseModel):
    extracted_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

class HandleExtractionResultToolOutput(BaseModel):
    text_content: str

class HandleExtractionResultTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="handle_extraction_result_tool",
            description="Handles the result from webpage extraction, returning details or an error/message string."
        )

    def _get_declaration(self):
         return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ExtractionResultInput.model_json_schema() 
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]: 
        tool_context.actions.skip_summarization = True # Ensure flow termination
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
        tool_context.actions.skip_summarization = True # Ensure flow termination
        try:
            input_data = CleanGitHubIssueTextToolInput.model_validate(args)
            cleaned_text = self.clean_text_logic(input_data.raw_text)
            logger.info(f"Tool: {self.name} - Cleaned text (first 100 chars): {cleaned_text[:100]}")
            return CleanGitHubIssueTextToolOutput(cleaned_text=cleaned_text).model_dump()
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error cleaning text: {e}. Args: {args}", exc_info=True)
            return CleanGitHubIssueTextToolOutput(cleaned_text=f"Error cleaning text: {e}").model_dump()

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
        self.llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL_NAME, temperature=0.1, google_api_key=API_KEY,)
        self.adk_context_for_llm = get_escaped_adk_context_for_llm()

    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ADKGuidanceToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]: 
        tool_context.actions.skip_summarization = True # Ensure flow termination
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

            prompt = f"""
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
"""
            logger.info(f"Tool: {self.name} - Prompting LLM with document (first 100 chars): {document_text[:100]}")
            response = await self.llm.ainvoke(prompt)
            guidance = response.content
            logger.info(f"Tool: {self.name} - Guidance received (first 100 chars): {guidance[:100]}")
            return ADKGuidanceToolOutput(guidance=guidance).model_dump()
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error generating guidance: {e}. Args: {args}", exc_info=True)
            return ADKGuidanceToolOutput(guidance=f"Error generating ADK guidance: {e}").model_dump()

class ExtractGitHubIssueDetailsToolInput(BaseModel): 
    url: str
    error: Optional[str] = None 

class ExtractGitHubIssueDetailsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="fetch_webpage_content_for_github_issue_tool",
            description="Fetches and extracts relevant content (like title, body, comments) from a given GitHub issue URL."
        )
        get_gemini_api_key_from_secret_manager() 
        self.browser_llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL_NAME, temperature=0)

    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ExtractGitHubIssueDetailsToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]: 
        tool_context.actions.skip_summarization = True # Ensure flow termination
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
        
import requests 
from typing import Dict, Any
from google.genai import types

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
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "owner": types.Schema(
                    type=types.Type.STRING,
                    default="google",
                    description="The owner of the GitHub repository (e.g., 'google').",
                ),
                "repo": types.Schema(
                    type=types.Type.STRING,
                    default="adk-python",
                    description="The name of the GitHub repository (e.g., 'adk-python').",
                ),
                "issue_number": types.Schema(
                    type=types.Type.INTEGER,
                    description="The number of the GitHub issue.",
                ),
            },
            required=["issue_number"], 
        ),
    )

  async def run_async(
      self, *, args: Dict[str, Any], tool_context: ToolContext
  ) -> Dict[str, Any]:
    tool_context.actions.skip_summarization = True # Add this line
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