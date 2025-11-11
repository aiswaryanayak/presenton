from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

# All existing endpoint imports
from api.v1.ppt.endpoints.slide_to_html import (
    SLIDE_TO_HTML_ROUTER,
    HTML_TO_REACT_ROUTER,
    HTML_EDIT_ROUTER,
    LAYOUT_MANAGEMENT_ROUTER,
)
from api.v1.ppt.endpoints.presentation import (
    PRESENTATION_ROUTER,
    generate_presentation_from_gemini,  # ✅ add Gemini wrapper import
)
from api.v1.ppt.endpoints.anthropic import ANTHROPIC_ROUTER
from api.v1.ppt.endpoints.google import GOOGLE_ROUTER
from api.v1.ppt.endpoints.openai import OPENAI_ROUTER
from api.v1.ppt.endpoints.files import FILES_ROUTER
from api.v1.ppt.endpoints.pptx_slides import PPTX_SLIDES_ROUTER, PPTX_FONTS_ROUTER
from api.v1.ppt.endpoints.pdf_slides import PDF_SLIDES_ROUTER
from api.v1.ppt.endpoints.fonts import FONTS_ROUTER
from api.v1.ppt.endpoints.icons import ICONS_ROUTER
from api.v1.ppt.endpoints.images import IMAGES_ROUTER
from api.v1.ppt.endpoints.ollama import OLLAMA_ROUTER
from api.v1.ppt.endpoints.outlines import OUTLINES_ROUTER
from api.v1.ppt.endpoints.slide import SLIDE_ROUTER


# ---------------------------------------------------------------- #
# ✅ Main API router
# ---------------------------------------------------------------- #
API_V1_PPT_ROUTER = APIRouter(prefix="/api/v1/ppt")

# ---------------------------------------------------------------- #
# ✅ Existing integrations
# ---------------------------------------------------------------- #
API_V1_PPT_ROUTER.include_router(FILES_ROUTER)
API_V1_PPT_ROUTER.include_router(FONTS_ROUTER)
API_V1_PPT_ROUTER.include_router(OUTLINES_ROUTER)
API_V1_PPT_ROUTER.include_router(PRESENTATION_ROUTER)
API_V1_PPT_ROUTER.include_router(PPTX_SLIDES_ROUTER)
API_V1_PPT_ROUTER.include_router(SLIDE_ROUTER)
API_V1_PPT_ROUTER.include_router(SLIDE_TO_HTML_ROUTER)
API_V1_PPT_ROUTER.include_router(HTML_TO_REACT_ROUTER)
API_V1_PPT_ROUTER.include_router(HTML_EDIT_ROUTER)
API_V1_PPT_ROUTER.include_router(LAYOUT_MANAGEMENT_ROUTER)
API_V1_PPT_ROUTER.include_router(IMAGES_ROUTER)
API_V1_PPT_ROUTER.include_router(ICONS_ROUTER)
API_V1_PPT_ROUTER.include_router(OLLAMA_ROUTER)
API_V1_PPT_ROUTER.include_router(PDF_SLIDES_ROUTER)
API_V1_PPT_ROUTER.include_router(OPENAI_ROUTER)
API_V1_PPT_ROUTER.include_router(ANTHROPIC_ROUTER)
API_V1_PPT_ROUTER.include_router(GOOGLE_ROUTER)
API_V1_PPT_ROUTER.include_router(PPTX_FONTS_ROUTER)


# ---------------------------------------------------------------- #
# ✅ Add Gemini AI endpoint (the missing one)
# ---------------------------------------------------------------- #
class GeminiDeckRequest(BaseModel):
    content: str
    n_slides: Optional[int] = 10
    template: Optional[str] = "modern"
    export_as: Optional[str] = "pptx"


@API_V1_PPT_ROUTER.post("/from_gemini")
async def from_gemini(request: GeminiDeckRequest):
    """
    Custom Gemini → Pitch Deck generation endpoint.
    Used by the AI Fundraising Support frontend.
    """
    try:
        result = await generate_presentation_from_gemini(
            content=request.content,
            n_slides=request.n_slides,
            export_as=request.export_as,
            template=request.template,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini rendering failed: {str(e)}")

