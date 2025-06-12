"""
This __init__.py file makes the reusable tools and their associated
Pydantic models directly importable from the 'expert_agents.tools' subpackage.

This provides a clean and organized way for agents to access the building
blocks they need to perform their functions. The __all__ list defines the
public API of this subpackage.
"""

from .adk_guidance_tool import (
    ADKGuidanceTool,
    ADKGuidanceToolInput,
    ADKGuidanceToolOutput,
)
from .clean_github_issue_text_tool import (
    CleanGitHubIssueTextTool,
    CleanGitHubIssueTextToolInput,
    CleanGitHubIssueTextToolOutput,
)
from .construct_github_url_tool import (
    ConstructGitHubUrlTool,
    ConstructGitHubUrlToolInput,
    ConstructGitHubUrlToolOutput,
)
from .extract_github_issue_details_tool import (
    ExtractGitHubIssueDetailsTool,
    ExtractGitHubIssueDetailsToolInput,
    ExtractionResultInput,
)
from .github_issue_tool import (
    GetGithubIssueDescriptionTool,
    get_github_issue_description,
)
from .github_utils import (
    get_github_pat_from_secret_manager,
    BOILERPLATE_STRINGS_TO_REMOVE,
)
from .handle_extraction_result_tool import (
    HandleExtractionResultTool,
    HandleExtractionResultToolOutput,
)
from .marp_document_tools import (
    generate_pdf_from_markdown_with_gcs,
    generate_html_slides_from_markdown_with_gcs,
    generate_pptx_slides_from_markdown_with_gcs,
)
from .marp_utils import DOC_LINK_STATE_KEY
from .mermaid_to_png_and_upload_tool import (
    MermaidToPngAndUploadTool,
    GCS_LINK_STATE_KEY,
    mermaid_gcs_tool_instance,
)
from .prepare_document_content_tool import (
    PrepareDocumentContentTool,
    PrepareDocumentContentToolInput,
)

__all__ = [
    # --- ADK Guidance Tool ---
    "ADKGuidanceTool",
    "ADKGuidanceToolInput",
    "ADKGuidanceToolOutput",
    # --- GitHub Processing Tools (Alternative Browser-based Flow) ---
    "CleanGitHubIssueTextTool",
    "CleanGitHubIssueTextToolInput",
    "CleanGitHubIssueTextToolOutput",
    "ConstructGitHubUrlTool",
    "ConstructGitHubUrlToolInput",
    "ConstructGitHubUrlToolOutput",
    "ExtractGitHubIssueDetailsTool",
    "ExtractGitHubIssueDetailsToolInput",
    "ExtractionResultInput",
    "HandleExtractionResultTool",
    "HandleExtractionResultToolOutput",
    # --- GitHub Processing Tools (Primary API-based Flow) ---
    "GetGithubIssueDescriptionTool",
    "get_github_issue_description",
    # --- GitHub Utils ---
    "get_github_pat_from_secret_manager",
    "BOILERPLATE_STRINGS_TO_REMOVE",
    # --- Document Generation Tools & Utils ---
    "generate_pdf_from_markdown_with_gcs",
    "generate_html_slides_from_markdown_with_gcs",
    "generate_pptx_slides_from_markdown_with_gcs",
    "DOC_LINK_STATE_KEY",
    "PrepareDocumentContentTool",
    "PrepareDocumentContentToolInput",
    # --- Mermaid Diagram Tools & Utils ---
    "MermaidToPngAndUploadTool",
    "GCS_LINK_STATE_KEY",
    "mermaid_gcs_tool_instance",
]
