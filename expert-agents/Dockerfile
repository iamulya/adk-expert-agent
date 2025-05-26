# iamulya-adk-expert-agent/expert-agents/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies:
# - For Node.js (required by mermaid-cli and marp-cli)
# - For Puppeteer (used by mermaid-cli)
# - curl, gnupg for adding NodeSource repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    # Puppeteer/Chromium dependencies. This list is comprehensive and similar
    # to what Playwright's --with-deps would install for Chromium.
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    wget \
    xdg-utils \
    # Install Node.js (e.g., version 20.x)
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    # Clean up apt cache
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Tell Puppeteer to skip downloading Chrome during npm install, we'll try to control it or use its default.
# This is often done if you provide your own Chrome. However, Marp/Mermaid bundle Puppeteer
# and expect Puppeteer to manage its browser. Let's let Puppeteer download its stuff.
# Ensure Puppeteer can download by setting a writable home directory for root.
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=false
ENV PUPPETEER_DOWNLOAD_HOST="https://storage.googleapis.com"
ENV PUPPETEER_DOWNLOAD_PATH="/root/.cache/puppeteer"
RUN mkdir -p /root/.cache/puppeteer && chown -R root:root /root/.cache

RUN npm install -g @mermaid-js/mermaid-cli @marp-team/marp-cli --unsafe-perm

ENV MERMAID_CLI_PATH=/usr/bin/mmdc

ENV CHROME_PATH=/root/.cache/puppeteer/chrome-headless-shell/linux-131.0.6778.204/chrome-headless-shell-linux64/chrome-headless-shell

# Global ENV vars for Puppeteer/Chrome behavior at RUNTIME
ENV PUPPETEER_NO_SANDBOX=true
ENV CHROME_LAUNCHER_ARGS="--no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage"

# Verify CLIs (CHROME_PATH should now be set if detection worked)
RUN echo "Verifying CLIs with CHROME_PATH='${CHROME_PATH}'"
RUN ${MERMAID_CLI_PATH} --version # Mermaid uses its own puppeteer config mostly
RUN marp --version # Marp should now pick up CHROME_PATH

WORKDIR /app

COPY expert-agents/test.mmd /app/test.mmd
COPY expert-agents/test.md /app/test.md
COPY expert-agents/puppeteer-config.json /app/puppeteer-config.json

# mmdc test (uses its -p config for sandbox, CHROME_PATH is less critical for it)
RUN ${MERMAID_CLI_PATH} -p /app/puppeteer-config.json -i /app/test.mmd -o /app/test.png && \
    echo "Mermaid CLI (mmdc) PNG generation test successful" && \
    rm /app/test.png && \
    echo "Removed /app/test.png"

# Marp test - should now use the globally set CHROME_PATH
RUN marp /app/test.md --pdf --allow-local-files -o /app/test.pdf && \
    echo "Marp CLI PDF generation test successful" && \
    rm /app/test.pdf && \
    echo "Removed /app/test.pdf"

RUN marp /app/test.md --html --allow-local-files -o /app/test.marp.html && \
    echo "Marp CLI HTML generation test successful" && \
    rm /app/test.marp.html && \
    echo "Removed /app/test.marp.html"

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
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000", "--allow_origins", "https://adk-expert-frontend-1033878884505.us-central1.run.app", "."]