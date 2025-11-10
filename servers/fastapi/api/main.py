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
from api.v1.ppt.endpoints.presentation import PRESENTATION_ROUTER  # ✅ fixed

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
app.include_router(PRESENTATION_ROUTER)  # ✅ add this

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
# ✅ Root endpoint
# ---------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "message": "✅ Presenton Gemini Visual Generator is live!",
        "status": "running",
        "endpoints": [
            "/presentation/generate?link=",
            "/api/v1/ppt/from_gemini (optional)",
        ],
    }
