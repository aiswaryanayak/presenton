# Use a lightweight Python base image
FROM python:3.11-slim-bookworm

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    libreoffice \
    fontconfig \
    chromium \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (for Next.js)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR /app  

# ----------------------------
# Environment variables
# ----------------------------
ENV APP_DATA_DIRECTORY=/app_data
ENV TEMP_DIRECTORY=/tmp/presenton
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# ----------------------------
# Install Ollama (optional local LLM runtime)
# ----------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh || echo "Skipping Ollama install"

# ----------------------------
# Install Python dependencies (cleaned)
# ----------------------------
RUN pip install --no-cache-dir aiohttp aiomysql aiosqlite asyncpg fastapi[standard] \
    pathvalidate pdfplumber sqlmodel \
    anthropic google-genai openai fastmcp dirtyjson

# Optional: docling for document parsing (CPU-only build)
RUN pip install --no-cache-dir docling --extra-index-url https://download.pytorch.org/whl/cpu

# ----------------------------
# Install dependencies & build Next.js frontend
# ----------------------------
WORKDIR /app/servers/nextjs
COPY servers/nextjs/package.json servers/nextjs/package-lock.json ./
RUN npm ci

COPY servers/nextjs/ /app/servers/nextjs/
RUN npm run build

# ----------------------------
# Copy FastAPI backend
# ----------------------------
WORKDIR /app
COPY servers/fastapi/ ./servers/fastapi/
COPY start.js LICENSE NOTICE ./

# ----------------------------
# Nginx setup
# ----------------------------
COPY nginx.conf /etc/nginx/nginx.conf

# ----------------------------
# Expose port
# ----------------------------
EXPOSE 80

# ----------------------------
# Start both servers
# ----------------------------
CMD ["bash", "-c", "service nginx start && uvicorn servers.fastapi.api.main:app --host 0.0.0.0 --port 8000"]

CMD ["node", "/app/start.js"]
