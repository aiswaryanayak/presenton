# servers/fastapi/api/v1/ppt/endpoints/presentation.py
import asyncio
import json
import random
import re
import traceback
import uuid
from datetime import datetime
from typing import Any, List, Optional, Type

import dirtyjson
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

# --- Internal imports ---
from servers.fastapi.models.api_error_model import APIErrorModel
from servers.fastapi.models.generate_presentation_request import GeneratePresentationRequest
from servers.fastapi.models.presentation_and_path import PresentationPathAndEditPath
from servers.fastapi.models.presentation_outline_model import (
    PresentationOutlineModel,
    SlideOutlineModel,
)
from servers.fastapi.enums.tone import Tone
from servers.fastapi.enums.verbosity import Verbosity
from servers.fastapi.models.presentation_with_slides import PresentationWithSlides
from servers.fastapi.models.sql.slide import SlideModel
from servers.fastapi.services.database import get_async_session
from servers.fastapi.services.image_generation_service import ImageGenerationService
from servers.fastapi.services.pptx_presentation_creator import PptxPresentationCreator
from servers.fastapi.models.sql.presentation import PresentationModel
from servers.fastapi.utils.asset_directory_utils import get_images_directory
from servers.fastapi.utils.export_utils import export_presentation
from servers.fastapi.utils.llm_calls.generate_presentation_outlines import generate_ppt_outline
from servers.fastapi.utils.llm_calls.generate_presentation_structure import generate_presentation_structure
from servers.fastapi.utils.llm_calls.generate_slide_content import get_slide_content_from_type_and_outline
from servers.fastapi.utils.ppt_utils import get_presentation_title_from_outlines
from servers.fastapi.utils.process_slides import process_slide_and_fetch_assets

# ✅ Force use of Hybrid layout
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)

PRESENTATION_ROUTER = APIRouter(prefix="/presentation", tags=["Presentation"])


# -------------------------
# Helpers
# -------------------------
def coerce_enum(enum_cls: Type[Any], value: Any, default: Any):
    """Safely coerce a value to an enum."""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception:
        pass
    try:
        if isinstance(value, str):
            return enum_cls[value.upper()]
    except Exception:
        pass
    return default


def _heuristic_create_slides_from_text(raw_text: str, n_slides_target: int) -> List[dict]:
    """Fallback heuristic: create slides from text if parsing fails."""
    t = (raw_text or "").strip()
    t = re.sub(r"```(?:json)?", "", t)
    segments = [s.strip() for s in re.split(r"\n-{3,}\n|\n{2,}", t) if s.strip()]
    if not segments:
        segments = [t] if t else [""]

    if len(segments) >= n_slides_target:
        chosen = segments[:n_slides_target]
    else:
        chosen = segments[:]
        for seg in segments:
            if len(chosen) >= n_slides_target:
                break
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', seg) if s.strip()]
            for s in sentences:
                if len(chosen) >= n_slides_target:
                    break
                if s not in chosen:
                    chosen.append(s)
        chosen = chosen[:n_slides_target]

    slides = []
    for seg in chosen:
        lines = [l.strip() for l in seg.splitlines() if l.strip()]
        title = lines[0] if lines else (seg[:40] if seg else "Untitled")
        content_lines = lines[1:] if len(lines) > 1 else []
        if not content_lines:
            bullets = [s.strip() for s in re.split(r'(?<=[.!?])\s+', seg) if s.strip()]
            bullets = bullets[:3] if bullets else [seg[:120]]
            content_lines = bullets
        slides.append({
            "title": title[:120],
            "subtitle": "",
            "content": content_lines,
            "visual_theme": "hybrid gradient, bold headings, image placeholders",
            "speaker_notes": "Auto-generated presenter notes.",
        })
    return slides


# -------------------------
# CRUD
# -------------------------
@PRESENTATION_ROUTER.get("/all", response_model=List[PresentationWithSlides])
async def get_all_presentations(sql_session: AsyncSession = Depends(get_async_session)):
    results = await sql_session.execute(
        select(PresentationModel, SlideModel)
        .join(SlideModel, (SlideModel.presentation == PresentationModel.id) & (SlideModel.index == 0))
        .order_by(PresentationModel.created_at.desc())
    )
    rows = results.all()
    return [
        PresentationWithSlides(**presentation.model_dump(), slides=[slide])
        for presentation, slide in rows
    ]


# -------------------------
# Generation Handler
# -------------------------
async def generate_presentation_handler(
    request: GeneratePresentationRequest,
    presentation_id: uuid.UUID,
    sql_session: AsyncSession,
):
    try:
        using_slides_markdown = bool(request.slides_markdown)
        if using_slides_markdown:
            request.n_slides = len(request.slides_markdown)

        # 1️⃣ Generate outlines
        if not using_slides_markdown:
            n_slides = request.n_slides or 10
            text_data = ""

            async for chunk in generate_ppt_outline(
                request.content,
                n_slides,
                request.language,
                "",
                request.tone.value,
                request.verbosity.value,
                request.instructions,
                request.include_title_slide,
                request.web_search,
            ):
                text_data += json.dumps(chunk) if isinstance(chunk, dict) else str(chunk)

            try:
                outlines_json = json.loads(text_data)
            except Exception:
                try:
                    outlines_json = dirtyjson.loads(text_data)
                except Exception:
                    outlines_json = {"slides": _heuristic_create_slides_from_text(text_data, n_slides)}

            if not isinstance(outlines_json, dict) or "slides" not in outlines_json:
                outlines_json = {"slides": _heuristic_create_slides_from_text(text_data, n_slides)}

            presentation_outlines = PresentationOutlineModel(**outlines_json)
            total_slides = len(presentation_outlines.slides)
        else:
            presentation_outlines = PresentationOutlineModel(
                slides=[SlideOutlineModel(content=slide) for slide in request.slides_markdown]
            )
            total_slides = len(request.slides_markdown)

        # 2️⃣ Load Hybrid Layout
        layout_model = PresentationLayoutModel()
        presentation_structure = await generate_presentation_structure(
            presentation_outline=presentation_outlines,
            presentation_layout=layout_model,
            instructions=request.instructions,
        )

        # 3️⃣ Create Presentation DB model
        presentation = PresentationModel(
            id=presentation_id,
            content=request.content,
            n_slides=request.n_slides,
            language=request.language,
            title=get_presentation_title_from_outlines(presentation_outlines),
            outlines=presentation_outlines.model_dump(),
            layout=layout_model.model_dump(),
            structure=presentation_structure.model_dump(),
            tone=request.tone.value,
            verbosity=request.verbosity.value,
            instructions=request.instructions,
        )

        image_service = ImageGenerationService(get_images_directory())
        slides, tasks = [], []

        for i, layout in enumerate(layout_model.slides):
            if i >= total_slides:
                break
            content = await get_slide_content_from_type_and_outline(
                layout,
                presentation_outlines.slides[i],
                request.language,
                request.tone.value,
                request.verbosity.value,
                request.instructions,
            )
            slide = SlideModel(
                presentation=presentation_id,
                layout_group=layout_model.name,
                layout=layout.id,
                index=i,
                speaker_note=content.get("__speaker_note__"),
                content=content,
            )
            slides.append(slide)
            tasks.append(process_slide_and_fetch_assets(image_service, slide))

        generated_assets = [asset for group in await asyncio.gather(*tasks) for asset in group]

        sql_session.add(presentation)
        sql_session.add_all(slides)
        sql_session.add_all(generated_assets)
        await sql_session.commit()

        presentation_path = await export_presentation(
            presentation_id,
            presentation.title or str(uuid.uuid4()),
            request.export_as,
        )
        return PresentationPathAndEditPath(
            **presentation_path.model_dump(),
            edit_path=f"/presentation?id={presentation_id}",
        )

    except Exception as e:
        print("🔥 DEBUG: Presentation generation failed:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate presentation: {str(e)}")


# -------------------------
# Public Endpoint
# -------------------------
@PRESENTATION_ROUTER.post("/generate", response_model=PresentationPathAndEditPath)
async def generate_presentation_sync(
    request: GeneratePresentationRequest,
    sql_session: AsyncSession = Depends(get_async_session),
):
    return await generate_presentation_handler(request, uuid.uuid4(), sql_session)


# -------------------------
# Gemini Wrapper (always hybrid)
# -------------------------
async def generate_presentation_from_gemini(**kwargs):
    tone_enum = coerce_enum(Tone, kwargs.get("tone", "default"), Tone.DEFAULT)
    verbosity_enum = coerce_enum(Verbosity, kwargs.get("verbosity", "standard"), Verbosity.STANDARD)

    request = GeneratePresentationRequest(
        content=kwargs.get("content", ""),
        n_slides=kwargs.get("n_slides", 10),
        export_as=kwargs.get("export_as", "pptx"),
        language=kwargs.get("language", "English"),
        template="hybrid",  # ✅ always hybrid
        tone=tone_enum,
        verbosity=verbosity_enum,
        instructions=kwargs.get("instructions", ""),
        include_title_slide=True,
        include_table_of_contents=False,
        web_search=False,
    )

    async for session in get_async_session():
        return await generate_presentation_handler(request, uuid.uuid4(), session)


generate_presentation = generate_presentation_from_gemini
