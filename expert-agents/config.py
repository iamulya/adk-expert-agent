# iamulya-adk-expert-agent/expert-agents/config.py
# Centralized configuration for the ADK Expert Agent
from google.cloud import secretmanager
import os

# The default model name to be used by agents and tools
DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"#"gemini-2.5-flash-preview-05-20""gemini-2.5-pro-preview-05-06"
PRO_MODEL_NAME = "gemini-2.5-pro-preview-05-06" # The model name for the pro version

_GEMINI_API_KEY = None

# GCS Configuration - GET THESE FROM .env
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME_FOR_DOCS")
GCS_PROJECT_ID_FOR_BUCKET = os.getenv("GCS_PROJECT_ID_FOR_DOCS_BUCKET") # Project where the bucket resides
GCS_SIGNED_URL_SA_EMAIL = os.getenv("GCS_SIGNED_URL_SA_EMAIL") # SA to impersonate for signing URLs
SIGNED_URL_EXPIRATION_SECONDS = int(os.getenv("SIGNED_URL_EXPIRATION_SECONDS", "3600")) # Default 1 hour

def get_gemini_api_key_from_secret_manager() -> str:
    global _GEMINI_API_KEY
    if _GEMINI_API_KEY:
        return _GEMINI_API_KEY
    project_number = os.getenv("GCP_PROJECT_NUMBER")
    secret_id = os.getenv("GEMINI_API_KEY_SECRET_ID")
    version_id = os.getenv("GEMINI_API_KEY_SECRET_VERSION", "1") 
    if not project_number or not secret_id:
        raise ValueError("GCP_PROJECT_NUMBER and GEMINI_API_KEY_SECRET_ID must be set in .env")
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

API_KEY = get_gemini_api_key_from_secret_manager()