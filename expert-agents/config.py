# Centralized configuration for the ADK Expert Agent
"""
This module centralizes the configuration for the ADK Expert Agent.

It handles loading environment variables, fetching secrets from Google Cloud Secret Manager,
and making them available to other parts of the application. This approach avoids
hardcoding sensitive information and allows for easy configuration changes.
"""

from google.cloud import secretmanager
import os

# The default model name to be used by agents and tools
# Using a less powerful model by default can save costs for simpler tasks.
DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-05-20"  # "gemini-2.5-flash-preview-04-17"#"gemini-2.5-pro-preview-05-06"
# A more powerful model for tasks that require higher quality generation, like creating documents or complex diagrams.
PRO_MODEL_NAME = "gemini-2.5-pro-preview-06-05"  # The model name for the pro version

# Global cache for the Gemini API key to avoid multiple fetches.
_GEMINI_API_KEY = None

# GCS Configuration - these values are loaded from the .env file.
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME_FOR_DOCS")
GCS_PROJECT_ID_FOR_BUCKET = os.getenv(
    "GCS_PROJECT_ID_FOR_DOCS_BUCKET"
)  # Project where the bucket resides
GCS_SIGNED_URL_SA_EMAIL = os.getenv(
    "GCS_SIGNED_URL_SA_EMAIL"
)  # SA to impersonate for signing URLs
SIGNED_URL_EXPIRATION_SECONDS = int(
    os.getenv("SIGNED_URL_EXPIRATION_SECONDS", "3600")
)  # Default 1 hour

# Path to the Mermaid CLI executable. Defaults to 'mmdc' assuming it's in the system's PATH.
MERMAID_CLI_PATH = os.getenv("MERMAID_CLI_PATH", "mmdc")


def get_gemini_api_key_from_secret_manager() -> str:
    """
    Fetches the Gemini API key from Google Cloud Secret Manager.

    It uses environment variables (GCP_PROJECT_NUMBER, GEMINI_API_KEY_SECRET_ID)
    to construct the secret's resource name. The fetched key is cached in a global
    variable to avoid repeated API calls during the application's lifecycle.
    It also sets the GOOGLE_API_KEY environment variable, which some Google
    libraries use automatically.

    Raises:
        ValueError: If required environment variables for secret access are not set.
        Exception: If the secret cannot be fetched from Secret Manager.

    Returns:
        The Gemini API key as a string.
    """
    global _GEMINI_API_KEY
    if _GEMINI_API_KEY:
        return _GEMINI_API_KEY
    project_number = os.getenv("GCP_PROJECT_NUMBER")
    secret_id = os.getenv("GEMINI_API_KEY_SECRET_ID")
    version_id = os.getenv("GEMINI_API_KEY_SECRET_VERSION", "1")
    if not project_number or not secret_id:
        raise ValueError(
            "GCP_PROJECT_NUMBER and GEMINI_API_KEY_SECRET_ID must be set in .env"
        )
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_number}/secrets/{secret_id}/versions/{version_id}"
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


# Fetch the API key on module load. This ensures the application fails fast
# if the key is not available, rather than failing on the first API call.
API_KEY = get_gemini_api_key_from_secret_manager()
