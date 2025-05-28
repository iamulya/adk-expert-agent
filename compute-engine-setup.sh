#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipestatus: The return value of a pipeline is the status of
# the last command to exit with a non-zero status.
set -o pipefail

echo "--- Starting Python 3.12, uv, Node.js, Angular CLI, Mermaid & Marp Setup for Ubuntu ---"

# 1. Update package lists and upgrade existing packages
echo "--- Updating system packages ---"
sudo apt update && sudo apt upgrade -y

# 2. Install prerequisites
echo "--- Installing prerequisite packages (software-properties-common, curl, build-essential) ---"
sudo apt install -y software-properties-common curl build-essential

# 3. Add deadsnakes PPA for Python 3.12
echo "--- Adding deadsnakes PPA for Python 3.12 ---"
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# 4. Install Python 3.12
echo "--- Installing Python 3.12, python3.12-dev, and python3.12-venv ---"
sudo apt install -y python3.12 python3.12-dev python3.12-venv

# 5. Ensure pip for Python 3.12 is installed and upgraded
echo "--- Ensuring pip for Python 3.12 is installed and upgraded ---"
python3.12 -m ensurepip --upgrade
python3.12 -m pip install --upgrade pip

# 6. Install uv (universal Python package installer/resolver)
echo "--- Installing uv ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
# The install script typically adds ~/.cargo/bin to PATH in .bashrc or similar.
# The user will likely need to run 'source ~/.bashrc' or open a new terminal.

# 7. Add Nodesource PPA for Node.js LTS (currently 20.x)
echo "--- Setting up Nodesource repository for Node.js LTS (20.x) ---"
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
# For a specific version like 20.x, use:
# curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

# 8. Install Node.js and npm
echo "--- Installing Node.js and npm ---"
sudo apt-get install -y nodejs

# 9. Install common dependencies for headless Chromium (used by Puppeteer for Marp CLI)
echo "--- Installing common dependencies for headless Chromium (Puppeteer) ---"
# Puppeteer bundles Chromium, but these system libraries can prevent sandbox issues.
# This list is a common subset; more might be needed in some specific environments.
sudo apt-get install -y \
    libnss3 \
    libxss1 \
    libasound2 \
    libgbm1 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libfontconfig1 \
    libpango-1.0-0 \
    libcairo2 \
    libu2f-udev # Often needed for newer Chrome versions to avoid udev errors

# 10. Install Angular CLI globally
echo "--- Installing Angular CLI globally ---"
sudo npm install -g @angular/cli

# 11. Install Mermaid CLI (mmdc) globally
echo "--- Installing Mermaid CLI (mmdc) globally ---"
sudo npm install -g @mermaid-js/mermaid-cli

# 12. Install Marp CLI globally
echo "--- Installing Marp CLI globally ---"
sudo npm install -g @marp-team/marp-cli

echo "--- Installation Complete! Verifying versions... ---"

echo -n "Python 3.12: "
python3.12 --version

echo -n "pip (for Python 3.12): "
python3.12 -m pip --version

echo -n "uv: "
# Try to find uv, it's usually in ~/.cargo/bin
if command -v ~/.cargo/bin/uv &> /dev/null; then
    ~/.cargo/bin/uv --version
elif command -v uv &> /dev/null; then
    uv --version
else
    echo "uv command not found in PATH. Please add ~/.cargo/bin to your PATH."
    echo "Example: echo 'export PATH=\"\$HOME/.cargo/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
fi

echo -n "Node.js: "
node --version

echo -n "npm: "
npm --version

echo -n "Angular CLI: "
ng --version

echo -n "Mermaid CLI (mmdc): "
if command -v mmdc &> /dev/null; then
    mmdc --version
else
    echo "mmdc command not found. Installation might have failed or it's not in PATH (though global npm should handle this)."
fi

echo -n "Marp CLI: "
if command -v marp &> /dev/null; then
    marp --version
else
    echo "marp command not found. Installation might have failed or it's not in PATH (though global npm should handle this)."
fi

echo ""
echo "--- Setup Finished Successfully! ---"
echo ""
echo "Next steps for Python development:"
echo "1. Create a project directory: mkdir my_python_project && cd my_python_project"
echo "2. Create a virtual environment: python3.12 -m venv .venv"
echo "3. Activate it: source .venv/bin/activate"
echo "4. Install packages using uv: uv pip install <package_name>"
echo "   (or python3.12 -m pip install <package_name>)"
echo "5. Deactivate: deactivate"
echo ""
echo "Next steps for Angular development:"
echo "1. Create a new Angular project: ng new my-angular-app"
echo "2. Navigate into it: cd my-angular-app"
echo "3. Serve the application: ng serve"
echo ""
echo "Using Mermaid CLI (mmdc):"
echo "  mmdc -i input.mmd -o output.svg"
echo "  mmdc -i input.mmd -o output.png"
echo "  mmdc --help"
echo ""
echo "Using Marp CLI:"
echo "  marp your_presentation.md -o presentation.html"
echo "  marp your_presentation.md -o presentation.pdf"
echo "  marp --help"
echo ""
echo "IMPORTANT: You might need to open a new terminal or run 'source ~/.bashrc'"
echo "           for changes to your PATH (especially for 'uv') to take effect."