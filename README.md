# My ADK Expert Agent

This agent is an expert on Google's Agent Development Kit (ADK) version 0.5.0.
It can answer general questions about ADK and help find solutions to GitHub issues.

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

3.  **Install Playwright Browsers**:
    `browser-use` requires Playwright browsers. Install Chromium (recommended for compatibility):
    ```bash
    playwright install --with-deps chromium
    ```

4.  **Configure Environment Variables**:
    *   Copy `.env.example` to `.env`.
    *   Fill in your Google Cloud Project ID (`GCP_PROJECT_ID`) and the Secret ID (`GEMINI_API_KEY_SECRET_ID`) for your Gemini API key stored in Google Cloud Secret Manager.
    *   Ensure the service account or user running the agent has permissions to access this secret.

5.  **Google Cloud Authentication**:
    Ensure you are authenticated with Google Cloud, e.g., by running:
    ```bash
    gcloud auth application-default login
    ```

## Running the Agent

You can run this agent as a web application using the ADK CLI:

```bash
adk web .
```

This will start a local web server (usually on `http://127.0.0.1:8000`) where you can interact with the agent.

## How it Works

*   The agent uses `gemini-2.5-pro-preview-05-06` as its primary LLM. (This model name is centralized in expert-agents/config.py.)
*   API keys are fetched securely from Google Cloud Secret Manager.
*   It has access to the content of `google-adk-python-0.5-with-test.txt` for ADK-related queries.
*   If you ask about a GitHub issue, it will use the `browser-use` library (also powered by Gemini 2.5 Pro Preview) to fetch details from the GitHub issue page before formulating a response.
