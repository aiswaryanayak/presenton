FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    libreoffice \
    fontconfig \
    chromium \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 using NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app  

# Set environment variables
ENV APP_DATA_DIRECTORY=/app_data
ENV TEMP_DIRECTORY=/tmp/presenton
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install dependencies for FastAPI (removed chromadb ✅)
RUN pip install --no-cache-dir \
    aiohttp aiomysql aiosqlite asyncpg fastapi[standard] \
    pathvalidate pdfplumber sqlmodel \
    anthropic google-genai openai fastmcp dirtyjson \
    docling --extra-index-url https://download.pytorch.org/whl/cpu

# Install dependencies for Next.js
WORKDIR /app/servers/nextjs
COPY servers/nextjs/package.json servers/nextjs/package-lock.json ./
RUN npm ci

# Copy Next.js app
COPY servers/nextjs/ ./

# Build the Next.js app
RUN npm run build

# Go back to main directory
WORKDIR /app

# Copy FastAPI backend
COPY servers/fastapi/ ./servers/fastapi/
COPY start.js LICENSE NOTICE ./

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose HTTP port
EXPOSE 80

# Start both servers
CMD ["node", "/app/start.js"]

