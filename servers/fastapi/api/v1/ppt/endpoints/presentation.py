# servers/fastapi/api/v1/ppt/endpoints/presentation.py
import asyncio
from datetime import datetime
import json
import math
import os
import random
import traceback
import re
from typing import Annotated, List, Literal, Optional, Tuple, Type, Any
import dirtyjson
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
import uuid

# Internal constants / enums / models
from servers.fastapi.constants.presentation import DEFAULT_TEMPLATES
from servers.fastapi.enums.webhook_event import WebhookEvent
from servers.fastapi.models.api_error_model import APIErrorModel
from servers.fastapi.models.generate_presentation_request import GeneratePresentationRequest
from servers.fastapi.models.presentation_and_path import PresentationPathAndEditPath
from servers.fastapi.models.presentation_from_template import EditPresentationRequest
from servers.fastapi.models.presentation_outline_model import (
    PresentationOutlineModel,
    SlideOutlineModel,
)
from servers.fastapi.enums.tone import Tone
from servers.fastapi.enums.verbosity import Verbosity
from servers.fastapi.models.pptx_models import PptxPresentationModel
from servers.fastapi.models.presentation_layout import PresentationLayoutModel
from servers.fastapi.models.presentation_structure_model import PresentationStructureModel
from servers.fastapi.models.presentation_with_slides import PresentationWithSlides
from servers.fastapi.models.sql.template import TemplateModel

# Services and utils (absolute imports)
from servers.fastapi.services.documents_loader import DocumentsLoader
from servers.fastapi.services.webhook_service import WebhookService
from servers.fastapi.utils.get_layout_by_name import get_layout_by_name
from servers.fastapi.services.image_generation_service import ImageGenerationService
from servers.fastapi.utils.dict_utils import deep_update
from servers.fastapi.utils.export_utils import export_presentation
from servers.fastapi.utils.llm_calls.generate_presentation_outlines import generate_ppt_outline
from servers.fastapi.models.sql.slide import SlideModel
from servers.fastapi.models.sse_response import SSECompleteResponse, SSEErrorResponse, SSEResponse
from servers.fastapi.services.database import get_async_session
from servers.fastapi.services.temp_file_service import TEMP_FILE_SERVICE
from servers.fastapi.services.concurrent_service import CONCURRENT_SERVICE
from servers.fastapi.models.sql.presentation import PresentationModel
from servers.fastapi.services.pptx_presentation_creator import PptxPresentationCreator
from servers.fastapi.models.sql.async_presentation_generation_status import (
    AsyncPresentationGenerationTaskModel,
)
from servers.fastapi.utils.asset_directory_utils import get_exports_directory, get_images_directory
from servers.fastapi.utils.llm_calls.generate_presentation_structure import (
    generate_presentation_structure,
)
from servers.fastapi.utils.llm_calls.generate_slide_content import (
    get_slide_content_from_type_and_outline,
)
from servers.fastapi.utils.ppt_utils import (
    get_presentation_title_from_outlines,
    select_toc_or_list_slide_layout_index,
)
from servers.fastapi.utils.process_slides import (
    process_slide_add_placeholder_assets,
    process_slide_and_fetch_assets,
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
    t = raw_text.strip()
    t = re.sub(r"```(?:json)?", "", t)
    segments = [s.strip() for s in re.split(r"\n-{3,}\n|\n{2,}", t) if s.strip()]
    if not segments:
        segments = [t]
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
        if len(chosen) < n_slides_target:
            chunk_size = max(100, len(t) // max(1, n_slides_target))
            more = [t[i:i+chunk_size].strip() for i in range(0, len(t), chunk_size)]
            for m in more:
                if len(chosen) >= n_slides_target:
                    break
                if m and m not in chosen:
                    chosen.append(m)
        chosen = chosen[:n_slides_target]

    slides = []
    for seg in chosen:
        lines = [l.strip() for l in seg.splitlines() if l.strip()]
        title = lines[0] if lines else seg[:40]
        content_lines = lines[1:] if len(lines) > 1 else []
        if not content_lines:
            bullets = [s.strip() for s in re.split(r'(?<=[.!?])\s+', seg) if s.strip()]
            bullets = bullets[:3] if bullets else [seg[:120]]
            content_lines = bullets
        slides.append({
            "title": title[:120],
            "subtitle": "",
            "content": content_lines,
            "visual_theme": "modern gradient, hero image or abstract illustration",
            "speaker_notes": "Auto-generated presenter notes. Expand as needed.",
        })
    return slides


# -------------------------
# Basic CRUD
# -------------------------
@PRESENTATION_ROUTER.get("/all", response_model=List[PresentationWithSlides])
async def get_all_presentations(sql_session: AsyncSession = Depends(get_async_session)):
    query = (
        select(PresentationModel, SlideModel)
        .join(
            SlideModel,
            (SlideModel.presentation == PresentationModel.id) & (SlideModel.index == 0),
        )
        .order_by(PresentationModel.created_at.desc())
    )
    results = await sql_session.execute(query)
    rows = results.all()
    return [
        PresentationWithSlides(
            **presentation.model_dump(),
            slides=[first_slide],
        )
        for presentation, first_slide in rows
    ]


# -------------------------
# Core Generation Handler
# -------------------------
async def generate_presentation_handler(
    request: GeneratePresentationRequest,
    presentation_id: uuid.UUID,
    async_status: Optional[AsyncPresentationGenerationTaskModel],
    sql_session: AsyncSession = Depends(get_async_session),
):
    try:
        using_slides_markdown = bool(request.slides_markdown)
        if using_slides_markdown:
            request.n_slides = len(request.slides_markdown)

        if not using_slides_markdown:
            additional_context = ""
            if async_status:
                async_status.message = "Generating presentation outlines"
                async_status.updated_at = datetime.now()
                sql_session.add(async_status)
                await sql_session.commit()

            if request.files:
                loader = DocumentsLoader(file_paths=request.files)
                await loader.load_documents()
                if loader.documents:
                    additional_context = "\n\n".join(loader.documents)

            n_slides_to_generate = request.n_slides
            presentation_outlines_text = ""
            async for chunk in generate_ppt_outline(
                request.content,
                n_slides_to_generate,
                request.language,
                additional_context,
                request.tone.value,
                request.verbosity.value,
                request.instructions,
                request.include_title_slide,
                request.web_search,
            ):
                if isinstance(chunk, dict):
                    presentation_outlines_text += json.dumps(chunk)
                else:
                    presentation_outlines_text += str(chunk)

            # Parse safely
            try:
                presentation_outlines_json = json.loads(presentation_outlines_text)
            except Exception:
                try:
                    presentation_outlines_json = dict(dirtyjson.loads(presentation_outlines_text))
                except Exception:
                    slides_built = _heuristic_create_slides_from_text(presentation_outlines_text, n_slides_to_generate)
                    presentation_outlines_json = {"slides": slides_built}

            # --- Normalize Gemini content fields before model validation ---
            if isinstance(presentation_outlines_json, dict) and "slides" in presentation_outlines_json:
                normalized_slides = []
                for s in presentation_outlines_json.get("slides", []):
                    slide = dict(s)
                    c = slide.get("content")
                    if isinstance(c, list):
                        flattened = []
                        for item in c:
                            if isinstance(item, dict) and "text" in item:
                                flattened.append(item["text"])
                            elif isinstance(item, str):
                                flattened.append(item)
                            else:
                                flattened.append(str(item))
                        slide["content"] = flattened
                    elif isinstance(c, dict):
                        slide["content"] = [c.get("text", str(c))]
                    elif isinstance(c, (int, float)):
                        slide["content"] = [str(c)]
                    elif isinstance(c, str):
                        slide["content"] = [c]
                    elif c is None:
                        slide["content"] = ["(empty)"]
                    normalized_slides.append(slide)
                presentation_outlines_json["slides"] = normalized_slides
            else:
                slides_built = _heuristic_create_slides_from_text(presentation_outlines_text, n_slides_to_generate)
                presentation_outlines_json = {"slides": slides_built}

            # Normalize before model validation
if isinstance(presentation_outlines_json, dict) and "slides" in presentation_outlines_json:
    for slide in presentation_outlines_json["slides"]:
        # Ensure title and content are plain strings
        if not isinstance(slide.get("title"), str):
            slide["title"] = str(slide.get("title", ""))
        if isinstance(slide.get("content"), list):
            cleaned = []
            for c in slide["content"]:
                if isinstance(c, dict):
                    if "text" in c:
                        cleaned.append(c["text"])
                    else:
                        cleaned.append(str(c))
                elif isinstance(c, str):
                    cleaned.append(c)
                else:
                    cleaned.append(str(c))
            slide["content"] = cleaned
        elif isinstance(slide.get("content"), dict):
            slide["content"] = [str(slide["content"].get("text", ""))]
        elif slide.get("content") is None:
            slide["content"] = ["(empty)"]
else:
    # fallback heuristic if slides missing
    slides_built = _heuristic_create_slides_from_text(presentation_outlines_text, n_slides_to_generate)
    presentation_outlines_json = {"slides": slides_built}

presentation_outlines = PresentationOutlineModel(**presentation_outlines_json)

            total_outlines = len(presentation_outlines.slides)

        else:
            presentation_outlines = PresentationOutlineModel(
                slides=[SlideOutlineModel(content=slide) for slide in request.slides_markdown]
            )
            total_outlines = len(request.slides_markdown)

        layout_model = await get_layout_by_name(request.template)
        total_slide_layouts = len(layout_model.slides)
        if layout_model.ordered:
            presentation_structure = layout_model.to_presentation_structure()
        else:
            presentation_structure = await generate_presentation_structure(
                presentation_outline=presentation_outlines,
                presentation_layout=layout_model,
                instructions=request.instructions,
            )

        presentation_structure.slides = presentation_structure.slides[:total_outlines]
        for i in range(total_outlines):
            if presentation_structure.slides[i] >= total_slide_layouts:
                presentation_structure.slides[i] = random.randint(0, total_slide_layouts - 1)

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
        async_tasks, slides = [], []
        layouts = [layout_model.slides[idx] for idx in presentation_structure.slides]

        for i, layout in enumerate(layouts):
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
            async_tasks.append(process_slide_and_fetch_assets(image_service, slide))

        generated_assets = [a for lst in await asyncio.gather(*async_tasks) for a in lst]

        sql_session.add(presentation)
        sql_session.add_all(slides)
        sql_session.add_all(generated_assets)
        await sql_session.commit()

        presentation_and_path = await export_presentation(
            presentation_id, presentation.title or str(uuid.uuid4()), request.export_as
        )
        return PresentationPathAndEditPath(
            **presentation_and_path.model_dump(),
            edit_path=f"/presentation?id={presentation_id}",
        )

    except Exception as e:
        print("🔥 DEBUG: Presentation generation crashed:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Presentation generation failed: {str(e)}")


# -------------------------
# Public Endpoints
# -------------------------
@PRESENTATION_ROUTER.post("/generate", response_model=PresentationPathAndEditPath)
async def generate_presentation_sync(
    request: GeneratePresentationRequest,
    sql_session: AsyncSession = Depends(get_async_session),
):
    return await generate_presentation_handler(request, uuid.uuid4(), None, sql_session)


# -------------------------
# Gemini-compatible wrapper
# -------------------------
async def generate_presentation_from_gemini(**kwargs):
    tone_enum = coerce_enum(Tone, kwargs.get("tone", "default"), Tone.DEFAULT)
    verbosity_enum = coerce_enum(Verbosity, kwargs.get("verbosity", "standard"), Verbosity.STANDARD)

    request = GeneratePresentationRequest(
        content=kwargs.get("content", ""),
        n_slides=kwargs.get("n_slides", 10),
        export_as=kwargs.get("export_as", "pptx"),
        language=kwargs.get("language", "English"),
        template="modern",

        tone=tone_enum,
        verbosity=verbosity_enum,
        instructions=kwargs.get("instructions", ""),
        include_title_slide=kwargs.get("include_title_slide", True),
        include_table_of_contents=kwargs.get("include_table_of_contents", False),
        web_search=kwargs.get("web_search", False),
    )

    async for session in get_async_session():
        response = await generate_presentation_handler(request, uuid.uuid4(), None, session)
        return response


generate_presentation = generate_presentation_from_gemini
