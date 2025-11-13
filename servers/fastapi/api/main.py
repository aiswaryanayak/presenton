from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

# Presenton's existing imports
from api.lifespan import app_lifespan
from api.middlewares import UserConfigEnvUpdateMiddleware
from api.v1.ppt.router import API_V1_PPT_ROUTER
from api.v1.webhook.router import API_V1_WEBHOOK_ROUTER
from api.v1.mock.router import API_V1_MOCK_ROUTER
from api.v1.ppt.endpoints.presentation import (
    PRESENTATION_ROUTER,
    generate_presentation_from_gemini,  # Correct Gemini wrapper
)

# ---------------------------------------------------------------- #
# Initialize FastAPI App
# ---------------------------------------------------------------- #
app = FastAPI(lifespan=app_lifespan)

# ---------------------------------------------------------------- #
# Register Routers
# ---------------------------------------------------------------- #
app.include_router(API_V1_PPT_ROUTER)
app.include_router(API_V1_WEBHOOK_ROUTER)
app.include_router(API_V1_MOCK_ROUTER)
app.include_router(PRESENTATION_ROUTER)

# ---------------------------------------------------------------- #
# 🔥 FIXED CORS CONFIGURATION (100% Working)
# ---------------------------------------------------------------- #
# IMPORTANT:
# Do NOT use allow_origins + allow_origin_regex together.
# FastAPI ignores regex when allow_origins is set.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*",  # Allow all HTTPS origins (Vercel, Render, custom, preview URLs)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logging.info("🔥 CORS enabled using allow_origin_regex=https://.*")

# ---------------------------------------------------------------- #
# Middleware
# ---------------------------------------------------------------- #
app.add_middleware(UserConfigEnvUpdateMiddleware)

# ---------------------------------------------------------------- #
# Gemini Request Model
# ---------------------------------------------------------------- #
class GeminiDeckRequest(BaseModel):
    content: str
    n_slides: Optional[int] = 10
    template: Optional[str] = "modern"
    export_as: Optional[str] = "pptx"

# ---------------------------------------------------------------- #
# Gemini → Deck Endpoint
# ---------------------------------------------------------------- #
@app.post("/api/v1/ppt/from_gemini")
async def create_from_gemini(request: GeminiDeckRequest):
    try:
        result = await generate_presentation_from_gemini(
            content=request.content,
            n_slides=request.n_slides,
            export_as=request.export_as,
            template=request.template,
        )
        return result
    except Exception as e:
        logging.error(f"❌ Gemini rendering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini rendering failed: {str(e)}")

# ---------------------------------------------------------------- #
# Root Endpoint
# ---------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "message": "✅ Presenton Gemini Visual Generator is live!",
        "status": "running",
        "cors": "allow_origin_regex=https://.*",
        "endpoints": [
            "/api/v1/ppt/from_gemini",
            "/presentation/generate",
        ],
    }

# ---------------------------------------------------------------- #
# Quick CORS Test
# ---------------------------------------------------------------- #
@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS is working fine ✅"}

