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
    generate_presentation_from_gemini,  # ✅ correct Gemini import
)

# ---------------------------------------------------------------- #
# ✅ Initialize FastAPI app
# ---------------------------------------------------------------- #
app = FastAPI(lifespan=app_lifespan)

# ---------------------------------------------------------------- #
# ✅ Register existing routers
# ---------------------------------------------------------------- #
app.include_router(API_V1_PPT_ROUTER)
app.include_router(API_V1_WEBHOOK_ROUTER)
app.include_router(API_V1_MOCK_ROUTER)
app.include_router(PRESENTATION_ROUTER)

# ---------------------------------------------------------------- #
# ✅ Configure CORS (for Vercel + local dev + Render)
# ---------------------------------------------------------------- #
origins = [
    "https://ai-fundraising-support.vercel.app",   # production frontend
    "https://ai-fundraising-support.vercel.com",   # alt domain
    "http://localhost:3000",                       # local dev
    "https://presenton-1h7p.onrender.com",         # backend Render origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # ✅ allow all Vercel preview URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Debug log for verification in Render logs
logging.basicConfig(level=logging.INFO)
logging.info("✅ CORS enabled for: %s", origins)

# ---------------------------------------------------------------- #
# ✅ Middleware for environment config
# ---------------------------------------------------------------- #
app.add_middleware(UserConfigEnvUpdateMiddleware)

# ---------------------------------------------------------------- #
# ✅ Gemini → Deck endpoint
# ---------------------------------------------------------------- #
class GeminiDeckRequest(BaseModel):
    content: str
    n_slides: Optional[int] = 10
    template: Optional[str] = "modern"
    export_as: Optional[str] = "pptx"

@app.post("/api/v1/ppt/from_gemini")
async def create_from_gemini(request: GeminiDeckRequest):
    """
    Accepts Gemini-generated content and creates a full presentation deck.
    """
    try:
        # ✅ Use the correct wrapper that builds the internal request model
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
# ✅ Root endpoint
# ---------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "message": "✅ Presenton Gemini Visual Generator is live!",
        "status": "running",
        "allowed_origins": origins,
        "endpoints": [
            "/api/v1/ppt/from_gemini",
            "/presentation/generate",
        ],
    }

# ---------------------------------------------------------------- #
# ✅ Quick CORS test endpoint (optional)
# ---------------------------------------------------------------- #
@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS is working fine ✅"}
