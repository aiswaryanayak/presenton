from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Presenton's existing imports
from api.lifespan import app_lifespan
from api.middlewares import UserConfigEnvUpdateMiddleware
from api.v1.ppt.router import API_V1_PPT_ROUTER
from api.v1.webhook.router import API_V1_WEBHOOK_ROUTER
from api.v1.mock.router import API_V1_MOCK_ROUTER

# 👇 Import the presentation generator
from api.v1.ppt.endpoints.presentation import generate_presentation



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

# ---------------------------------------------------------------- #
# ✅ Configure CORS
# ---------------------------------------------------------------- #
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- #
# ✅ Middleware for environment config
# ---------------------------------------------------------------- #
app.add_middleware(UserConfigEnvUpdateMiddleware)


# ---------------------------------------------------------------- #
# 💡 NEW: Gemini Integration Endpoint
# ---------------------------------------------------------------- #
class GeminiDeckRequest(BaseModel):
    content: str
    n_slides: Optional[int] = 10
    template: Optional[str] = "modern"
    export_as: Optional[str] = "pptx"

@app.post("/api/v1/ppt/from_gemini")
async def create_from_gemini(request: GeminiDeckRequest):
    """
    Accepts Gemini-generated content and uses Presenton's rendering engine
    to create a full, visual deck (pptx/pdf) with chosen template.
    """
    try:
        result = await generate_presentation(
            content=request.content,
            n_slides=request.n_slides,
            export_as=request.export_as,
            template=request.template
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini rendering failed: {str(e)}")


# ---------------------------------------------------------------- #
# ✅ Root endpoint (optional)
# ---------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "message": "✅ Presenton Gemini Visual Generator is live!",
        "status": "running",
        "endpoints": [
            "/api/v1/ppt/from_gemini",
            "/api/v1/ppt/presentation/generate"
        ]
    }

