# My ADK Expert Agent

This agent is an expert on Google's Agent Development Kit (ADK) version 1.0.0.
It can answer general questions about ADK and help find solutions/guidance for GitHub issues.

## Directory Structure 

```
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
```

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
    If you plan to use tools that rely on `browser-use` (e.g., the `ExtractGitHubIssueDetailsTool`, though the primary GitHub issue processing in this agent no longer uses it by default), you'll need Playwright browsers. Install Chromium (recommended for compatibility):
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

