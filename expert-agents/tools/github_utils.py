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
