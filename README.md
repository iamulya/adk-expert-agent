# My ADK Expert Agent

This agent is an expert on Google's Agent Development Kit (ADK) version 1.0.0.
It can answer general questions about ADK and help find solutions/guidance for GitHub issues.

## Setup

1.  **Create a Python Virtual Environment**:
    It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    Make sure you have `uv` or `pip` installed.
    Using `uv` (recommended):
    ```bash
    pip install uv
    uv pip install -r requirements.txt  # Or directly from pyproject.toml if using uv sync
    # If pyproject.toml is set up for uv sync:
    # uv sync
    ```
    Using `pip`:
    ```bash
    pip install -e . 
    # This will install dependencies from pyproject.toml in editable mode.
    # Alternatively, generate a requirements.txt from pyproject.toml if needed.
    ```

3.  **Install Playwright Browsers (Optional for general use, if not using other browser-dependent tools)**:
    If you plan to use tools that rely on `browser-use` (though the primary GitHub issue processing in this agent no longer uses it), you'll need Playwright browsers. Install Chromium (recommended for compatibility):
    ```bash
    playwright install --with-deps chromium
    ```

4.  **Configure Environment Variables**:
    *   Copy `expert-agents/.env.example` to `expert-agents/.env`.
    *   Fill in your Google Cloud Project ID (`GCP_PROJECT_ID`).
    *   Fill in the Secret ID (`GEMINI_API_KEY_SECRET_ID`) for your Gemini API key stored in Google Cloud Secret Manager.
    *   Fill in the Secret ID (`GITHUB_API_PAT_SECRET_ID`) for your GitHub Personal Access Token (PAT) stored in Google Cloud Secret Manager. This PAT needs read access to public repositories (e.g., for fetching issue details).
    *   Ensure the service account or user running the agent has permissions to access these secrets.

5.  **Google Cloud Authentication**:
    Ensure you are authenticated with Google Cloud, e.g., by running:
    ```bash
    gcloud auth application-default login
    ```

## Running the Agent

You can run this agent as a web application using the ADK CLI:

```bash
adk web .