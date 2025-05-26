# iamulya-adk-expert-agent/expert-agents/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies:
# - For Node.js (required by mermaid-cli and marp-cli)
# - curl, gnupg for adding NodeSource repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    # Install Node.js (e.g., version 20.x)
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    # Clean up apt cache
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Mermaid CLI (mmdc) and Marp CLI globally using npm
# MARP_CLI_COMMAND in tools.py is "marp", so it expects 'marp' to be in PATH.
# MERMAID_CLI_PATH in config.py is "/usr/local/bin/mmdc".
RUN npm install -g @mermaid-js/mermaid-cli @marp-team/marp-cli
ENV MERMAID_CLI_PATH=/usr/local/bin/mmdc
# 'marp' will typically be at /usr/local/bin/marp, which should be in PATH

# Set the working directory in the container
WORKDIR /app

# Copy only the files necessary for dependency installation first
COPY pyproject.toml /app/
COPY README.md /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install .

# Copy the expert-agents application code into the container
COPY expert-agents /app/expert-agents/

# Expose the port the ADK API server will run on
EXPOSE 8000

# Command to run the ADK API server
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000", "--allow_origins", "*", "."]