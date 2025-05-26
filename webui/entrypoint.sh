#!/bin/sh
# iamulya-adk-expert-agent/webui/entrypoint.sh

set -e # Exit immediately if a command exits with a non-zero status.

CONFIG_FILE_PATH="/usr/share/nginx/html/assets/config/runtime-config.json"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE_PATH" ]; then
  echo "Error: $CONFIG_FILE_PATH not found. Creating a default one."
  mkdir -p "$(dirname "$CONFIG_FILE_PATH")"
  echo "{\"backendUrl\":\"${BACKEND_URL:-http://localhost:8000}\"}" > "$CONFIG_FILE_PATH"
fi

echo "Updating $CONFIG_FILE_PATH with backend URL: $BACKEND_URL"

# Use sed to replace the backendUrl value.
# This assumes the key "backendUrl" exists and its value is a string.
# It replaces "whatever_url" in "backendUrl": "whatever_url"
# with the content of the BACKEND_URL environment variable.
# Using a temporary file for safer sed operation.
TMP_CONFIG_FILE=$(mktemp)
sed 's|\("backendUrl":\s*"\)[^"]*\(".*\)|\1'"$BACKEND_URL"'\2|' "$CONFIG_FILE_PATH" > "$TMP_CONFIG_FILE" && mv "$TMP_CONFIG_FILE" "$CONFIG_FILE_PATH"

echo "Contents of $CONFIG_FILE_PATH after substitution:"
cat "$CONFIG_FILE_PATH"

# Execute the CMD from the Dockerfile (starts Nginx)
exec "$@"