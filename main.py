import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app

# Directory where your compiled Angular app is located inside the container
# This path must match the path in the Dockerfile
UI_BUILD_DIR = Path(__file__).parent / "dist"

# Directory where your ADK agent code is located
AGENTS_DIR = Path(__file__).parent / "expert_agents"

# --- FastAPI App Initialization ---
app = FastAPI(title="Expert ADK Agent")

# --- 1. Mount the ADK API routes ---
# We get the ADK API app and mount it under the /api prefix.
# This keeps your API routes separate from your UI routes.
adk_api_app = get_fast_api_app(
    agents_dir=str(AGENTS_DIR),
    session_db_url=os.environ.get("SESSION_DB_URL", ""),
    artifact_storage_uri=os.environ.get("ARTIFACT_STORAGE_URI", ""),
    web=False, # Important: We only want the API, not the built-in UI
    trace_to_cloud=os.environ.get("TRACE_TO_CLOUD", "false").lower() == "true",
)
app.mount("/api", adk_api_app)


# --- 2. Mount the custom Angular UI ---
# This serves your compiled Angular app's static files.
# The `html=True` argument makes it serve `index.html` for any path
# not found, which is essential for single-page applications (SPAs).
app.mount("/", StaticFiles(directory=UI_BUILD_DIR, html=True), name="ui")

print("Server configured to run custom UI and ADK API.")