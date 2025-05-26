# iamulya-adk-expert-agent/webui/Dockerfile

# ---- Build Stage ----
# Use an official Node.js LTS version for the build environment
FROM node:20-alpine AS build
LABEL stage=builder

WORKDIR /app

# Copy package.json and package-lock.json (if available) to leverage Docker cache
COPY webui/package.json webui/package-lock.json* ./

# Install dependencies. `npm ci` is preferred for CI/CD for reproducible builds.
# --legacy-peer-deps might be needed for some Angular versions/dependencies.
RUN npm install --legacy-peer-deps

# Copy the rest of the webui source code and auxiliary files for this stage
COPY webui/ ./
# Auxiliary files for the serve stage that need to be present in the build context
# COPY webui/nginx.conf /app/nginx.conf # Copied directly in serve stage from context for clarity
# COPY webui/entrypoint.sh /app/entrypoint.sh # Copied directly in serve stage from context for clarity

# Build the Angular application for production
# `npm run build` typically executes `ng build --configuration production`
# Output path is usually `dist/agent_framework_web` as per angular.json
RUN npm run build

# ---- Serve Stage ----
# Use a lightweight Nginx image to serve the static files
FROM nginx:1.25-alpine

# Nginx configuration
# Remove default Nginx welcome page and default configuration
RUN rm -rf /usr/share/nginx/html/* && \
    rm -f /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration file from the build context (webui directory)
COPY webui/nginx.conf /etc/nginx/nginx.conf

# Copy built static files from the build stage to Nginx's web root
COPY --from=build /app/dist/agent_framework_web /usr/share/nginx/html

# Copy the entrypoint script from the build context (webui directory) and make it executable
COPY webui/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port 80 for Nginx
EXPOSE 80

# Set the entrypoint to our custom script
ENTRYPOINT ["/entrypoint.sh"]

# Default command to start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]