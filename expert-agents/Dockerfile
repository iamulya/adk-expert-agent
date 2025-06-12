# iamulya-adk-expert-agent/expert-agents/Dockerfile
#
# This Dockerfile creates a container image for the ADK Expert Agent backend.
# It installs all necessary system dependencies, Python packages, and global Node.js tools.

# Use an official Python 3.12 slim image as the base. 'slim' is smaller than the full image.
FROM python:3.12-slim

# Set environment variables for a more efficient and predictable Python environment.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on \\
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies. This is done in a single RUN layer to reduce image size.
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
    # Tools for adding Node.js repository
    curl \\
    gnupg \\
    # Comprehensive list of dependencies for Puppeteer/Chromium, which is used by
    # both Marp CLI and Mermaid CLI for rendering.
    ca-certificates \\
    fonts-liberation \\
    libasound2 \\
    libatk-bridge2.0-0 \\
    libatk1.0-0 \\
    libcairo2 \\
    libcups2 \\
    libdbus-1-3 \\
    libexpat1 \\
    libfontconfig1 \\
    libgbm1 \\
    libgcc1 \\
    libglib2.0-0 \\
    libgtk-3-0 \\
    libnspr4 \\
    libnss3 \\
    libpango-1.0-0 \\
    libpangocairo-1.0-0 \\
    libstdc++6 \\
    libx11-6 \\
    libx11-xcb1 \\
    libxcb1 \\
    libxcomposite1 \\
    libxcursor1 \\
    libxdamage1 \\
    libxext6 \\
    libxfixes3 \\
    libxi6 \\
    libxrandr2 \\
    libxrender1 \\
    libxss1 \\
    libxtst6 \\
    lsb-release \\
    wget \\
    xdg-utils \\
    # Add the NodeSource repository for Node.js 20.x (LTS)
    && curl -fsSL <https://deb.nodesource.com/setup_20.x> | bash - \\
    # Install Node.js from the newly added repository
    && apt-get install -y nodejs \\
    # Clean up apt caches to reduce image size
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure Puppeteer environment variables.
# This ensures Puppeteer, when installed by npm, downloads its own managed version of Chromium.
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=false
ENV PUPPETEER_DOWNLOAD_HOST="<https://storage.googleapis.com>"
# Set a writable cache path for the root user.
ENV PUPPETEER_DOWNLOAD_PATH="/root/.cache/puppeteer"
RUN mkdir -p /root/.cache/puppeteer && chown -R root:root /root/.cache

# Install Mermaid and Marp CLIs globally using npm.
# --unsafe-perm is often needed when running npm as root in Docker.
RUN npm install -g @mermaid-js/mermaid-cli @marp-team/marp-cli --unsafe-perm

# Set the path to the Mermaid CLI executable.
ENV MERMAID_CLI_PATH=/usr/bin/mmdc
# Define the path to the Chromium executable that Puppeteer downloads.
# This helps Marp CLI find a compatible browser if it has trouble.
# Note: The version number might change with updates to puppeteer.
ENV CHROME_PATH=/root/.cache/puppeteer/chrome-headless-shell/linux-131.0.6778.204/chrome-headless-shell-linux64/chrome-headless-shell

# Set runtime environment variables for Puppeteer/Chrome.
# Running without a sandbox is required in many container environments.
ENV PUPPETEER_NO_SANDBOX=true
ENV CHROME_LAUNCHER_ARGS="--no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage"

# Verify that the CLIs are installed and executable.
RUN echo "Verifying CLIs..."
RUN ${MERMAID_CLI_PATH} --version
RUN marp --version

# Set the working directory for the application.
WORKDIR /app

# Copy test files and puppeteer config for build-time verification.
COPY expert-agents/test.mmd /app/test.mmd
COPY expert-agents/test.md /app/test.md
COPY expert-agents/puppeteer-config.json /app/puppeteer-config.json

# Run tests to ensure the CLI tools can generate files correctly within the container.
# This helps catch issues with dependencies or paths early.
RUN echo "Running build-time generation tests..."
RUN ${MERMAID_CLI_PATH} -p /app/puppeteer-config.json -i /app/test.mmd -o /app/test.png && \\
    echo "Mermaid CLI (mmdc) PNG generation test successful" && \\
    rm /app/test.png

RUN marp /app/test.md --pdf --allow-local-files -o /app/test.pdf && \\
    echo "Marp CLI PDF generation test successful" && \\
    rm /app/test.pdf

RUN marp /app/test.md --html --allow-local-files -o /app/test.marp.html && \\
    echo "Marp CLI HTML generation test successful" && \\
    rm /app/test.marp.html && \\
    echo "Build-time tests complete. Cleaning up test files."

# Copy project definition files to leverage Docker layer caching.
# Only these files are needed to install dependencies.
COPY pyproject.toml /app/
COPY README.md /app/

# Install Python dependencies using pip.
RUN pip install --upgrade pip
RUN pip install .

# Copy the rest of the application code into the container.
COPY expert-agents /app/expert-agents/

# Expose the port the ADK API server will run on.
EXPOSE 8000

# The command to run when the container starts.
# This starts the ADK API server, making the agent available.
# The origin is set to allow requests from a specific deployed frontend application.
# The '.' indicates the ADK application root is the current directory.
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000", "--allow_origins", "<https://adk-expert-frontend-1033878884505.us-central1.run.app>", "."]
