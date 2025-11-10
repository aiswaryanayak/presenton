FROM python:3.11-slim-bookworm

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    libreoffice \
    fontconfig \
    chromium \
    build-essential \
    nodejs npm \
    && apt-get clean

# Create working directory
WORKDIR /app

# Set environment variables
ENV APP_DATA_DIRECTORY=/app_data
ENV TEMP_DIRECTORY=/tmp/presenton
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PYTHONUNBUFFERED=1

# Install Ollama (optional, safe to keep)
RUN curl -fsSL https://ollama.com/install.sh | sh || true

# Copy FastAPI files first
COPY servers/fastapi/ /app/servers/fastapi/
COPY start.js LICENSE NOTICE ./

# Install Python dependencies (no chromadb)
RUN pip install --no-cache-dir \
    aiohttp aiomysql aiosqlite asyncpg fastapi[standard] sqlmodel \
    anthropic google-generativeai openai fastmcp dirtyjson pdfplumber pathvalidate \
    docling --extra-index-url https://download.pytorch.org/whl/cpu

# Install Node.js dependencies
WORKDIR /app/servers/nextjs
COPY servers/nextjs/package*.json ./
RUN npm install

# Build Next.js app
COPY servers/nextjs/ ./
RUN npm run build

WORKDIR /app

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["node", "start.js"]
