# SRT-AI Production Dockerfile
# Multi-stage build for smaller final image

FROM node:20-slim AS builder

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install all dependencies (including dev for potential build steps)
RUN npm ci

# Copy source code
COPY . .

# -------------------------------------------
# Production stage
FROM node:20-slim

# Install system dependencies
# - FFmpeg for audio/video processing
# - CA certificates for HTTPS requests
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r srtai && useradd -r -g srtai srtai

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production && npm cache clean --force

# Copy application code from builder
COPY --from=builder /app/server.js ./
COPY --from=builder /app/index.html ./

# Create temp directory with proper permissions
RUN mkdir -p /tmp/srt-ai && chown -R srtai:srtai /tmp/srt-ai /app

# Switch to non-root user
USER srtai

# Environment defaults
ENV NODE_ENV=production
ENV PORT=3000

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "fetch('http://localhost:3000/api/health').then(r => r.ok ? process.exit(0) : process.exit(1)).catch(() => process.exit(1))"

# Start the server
CMD ["node", "server.js"]
