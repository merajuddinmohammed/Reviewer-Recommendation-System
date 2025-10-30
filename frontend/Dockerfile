# ============================================================================
# Production Dockerfile - Reviewer Recommendation Frontend
# ============================================================================
# Multi-stage build: Node 20 builder → Nginx Alpine runner
# Deployment: Docker (Render Web Service) or Static Site (Render Static Site)
# ============================================================================

# ============================================================================
# Stage 1: Builder - Build React app with Node 20
# ============================================================================

FROM node:20-alpine AS builder

# Metadata
LABEL maintainer="Applied AI Assignment"
LABEL description="Reviewer Recommendation System - Frontend"
LABEL version="1.0"
LABEL stage="builder"

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
# Use npm ci for clean, reproducible installs
RUN npm ci --legacy-peer-deps

# Copy application source
COPY . .

# Build production bundle
# Vite outputs to dist/ directory
RUN npm run build

# Verify build output exists
RUN ls -la dist/ && \
    echo "Build completed successfully!"

# ============================================================================
# Stage 2: Runner - Serve with Nginx Alpine
# ============================================================================

FROM nginx:alpine AS runner

# Metadata
LABEL maintainer="Applied AI Assignment"
LABEL description="Reviewer Recommendation System - Frontend (Nginx)"
LABEL version="1.0"
LABEL stage="runner"

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built static files from builder stage
# Vite outputs to dist/ directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Verify files copied
RUN ls -la /usr/share/nginx/html/ && \
    echo "Static files copied successfully!"

# Create non-root user for security
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser && \
    chown -R appuser:appuser /usr/share/nginx/html && \
    chown -R appuser:appuser /var/cache/nginx && \
    chown -R appuser:appuser /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R appuser:appuser /var/run/nginx.pid

# Switch to non-root user
USER appuser

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# ============================================================================
# Build and Run Instructions
# ============================================================================
#
# Build:
#   docker build -t reviewer-recommender-frontend .
#
# Run locally:
#   docker run -p 3000:80 reviewer-recommender-frontend
#
# Run with custom backend URL:
#   docker build --build-arg VITE_API_BASE=https://api.example.com \
#     -t reviewer-recommender-frontend .
#   docker run -p 3000:80 reviewer-recommender-frontend
#
# Deploy to Render (Docker):
#   1. Push to GitHub
#   2. Create Web Service (Docker)
#   3. Set environment variables:
#      - PORT=10000 (Render assigns)
#   4. Deploy
#
# Deploy to Render (Static Site):
#   1. Push to GitHub
#   2. Create Static Site
#   3. Build command: npm ci && npm run build
#   4. Publish directory: dist
#   5. Set environment variables:
#      - VITE_API_BASE=https://your-backend.onrender.com
#
# ============================================================================
# Image Size Optimization
# ============================================================================
# - Multi-stage build (separates build from runtime)
# - Builder stage: node:20-alpine (~150 MB)
# - Runner stage: nginx:alpine (~40 MB)
# - Final image: ~45 MB (only nginx + static files)
# - No Node.js in final image (security benefit)
#
# ============================================================================
# Environment Variables
# ============================================================================
# Build-time variables (injected during docker build):
# - VITE_API_BASE: Backend API URL
#
# Runtime variables (for static site deployment):
# - Set in Render dashboard or .env file
#
# Note: Vite environment variables must be set at build time,
# not runtime. For runtime config, use build args or rebuild.
#
# ============================================================================
