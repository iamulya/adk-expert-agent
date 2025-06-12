import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from google.adk.cli.fast_api import get_fast_api_app

UI_BUILD_DIR = Path(__file__).parent / "dist" / "browser"
AGENTS_DIR = Path(__file__).parent #/ "expert_agents"

# --- FastAPI App Initialization ---
app = FastAPI(title="Expert ADK Agent")

# --- 1. Mount the ADK API routes ---
# We get the ADK API app and mount it under the /api prefix.
# This keeps your API routes separate from your UI routes.
adk_api_app = get_fast_api_app(
    agent_dir=str(AGENTS_DIR),
    session_db_url=os.environ.get("SESSION_DB_URL", ""),
    web=False, # Important: We only want the API, not the built-in UI
    trace_to_cloud=os.environ.get("TRACE_TO_CLOUD", "false").lower() == "true",
)
app.mount("/api", adk_api_app)


# --- 2. ADD REDIRECTS FOR A BETTER USER EXPERIENCE ---
# Redirect from the root to the UI base path
@app.get("/")
async def redirect_root_to_dev_ui():
    return RedirectResponse("/dev-ui/")

# Ensure a trailing slash for the UI base path
@app.get("/dev-ui")
async def redirect_dev_ui_add_slash():
    return RedirectResponse("/dev-ui/")


# --- 3. Mount the custom Angular UI AT THE CORRECT PATH ---
app.mount("/dev-ui/", StaticFiles(directory=UI_BUILD_DIR, html=True), name="ui")

print("Server configured to run custom UI and ADK API.")