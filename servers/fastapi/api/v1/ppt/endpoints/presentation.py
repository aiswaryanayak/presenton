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
    t = (raw_text or "").strip()
    t = re.sub(r"```(?:json)?", "", t)
    # Break on strong separators first, then on double newlines
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


@PRESENTATION_ROUTER.get("/{id}", response_model=PresentationWithSlides)
async def get_presentation(id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)):
    presentation = await sql_session.get(PresentationModel, id)
    if not presentation:
        raise HTTPException(404, "Presentation not found")
    slides = await sql_session.scalars(
        select(SlideModel)
        .where(SlideModel.presentation == id)
        .order_by(SlideModel.index)
    )
    return PresentationWithSlides(
        **presentation.model_dump(),
        slides=slides,
    )


@PRESENTATION_ROUTER.delete("/{id}", status_code=204)
async def delete_presentation(id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)):
    presentation = await sql_session.get(PresentationModel, id)
    if not presentation:
        raise HTTPException(404, "Presentation not found")

    await sql_session.delete(presentation)
    await sql_session.commit()


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

            n_slides_to_generate = request.n_slides or 10
            presentation_outlines_text = ""

            # Collect chunks from generator (LLM wrapper may stream strings/dicts)
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
                if isinstance(chunk, HTTPException):
                    raise chunk
                if isinstance(chunk, dict):
                    try:
                        presentation_outlines_text += json.dumps(chunk)
                    except Exception:
                        presentation_outlines_text += str(chunk)
                else:
                    presentation_outlines_text += str(chunk)

            # ---------- Parsing & robust fallback ----------
            presentation_outlines_json = None
            parse_exception = None
            try:
                presentation_outlines_json = json.loads(presentation_outlines_text)
            except Exception as e1:
                parse_exception = e1
                try:
                    presentation_outlines_json = dict(dirtyjson.loads(presentation_outlines_text))
                except Exception as e2:
                    parse_exception = e2
                    presentation_outlines_json = None

            # If parsed JSON doesn't contain slides, attempt regex extraction
            if not isinstance(presentation_outlines_json, dict) or "slides" not in presentation_outlines_json:
                try:
                    m = re.search(r'(\{.*"slides"\s*:\s*\[.*\].*?\})', presentation_outlines_text, re.DOTALL)
                    if m:
                        candidate = m.group(1)
                        presentation_outlines_json = json.loads(candidate)
                except Exception:
                    # ignore; we'll attempt heuristic below
                    presentation_outlines_json = None

            # Final fallback: build slides heuristically from raw text
            if not isinstance(presentation_outlines_json, dict) or "slides" not in presentation_outlines_json:
                print("⚠️ DEBUG: parsed outlines invalid or missing 'slides'. Falling back to heuristic.")
                if parse_exception:
                    print("⚠️ DEBUG parse exception:", repr(parse_exception))
                slides_built = _heuristic_create_slides_from_text(presentation_outlines_text, n_slides_to_generate)
                presentation_outlines_json = {"slides": slides_built}

            # Normalize slide content shapes for PresentationOutlineModel
            if isinstance(presentation_outlines_json, dict) and "slides" in presentation_outlines_json:
                normalized_slides = []
                for s in presentation_outlines_json.get("slides", []):
                    # s may be dict or string; coerce to dict
                    slide = s if isinstance(s, dict) else {"content": s, "title": None}
                    slide = dict(slide)  # shallow copy
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
                        # Gemini style content object -> extract text
                        text_val = c.get("text") or c.get("parts") or c.get("content")
                        if isinstance(text_val, list):
                            # flatten parts
                            flat = []
                            for p in text_val:
                                if isinstance(p, dict) and "text" in p:
                                    flat.append(p["text"])
                                elif isinstance(p, str):
                                    flat.append(p)
                                else:
                                    flat.append(str(p))
                            slide["content"] = flat
                        else:
                            slide["content"] = [str(text_val)] if text_val is not None else ["(empty)"]
                    elif isinstance(c, (int, float)):
                        slide["content"] = [str(c)]
                    elif isinstance(c, str):
                        slide["content"] = [c]
                    elif c is None:
                        # maybe slide has 'text' key instead of content
                        if "text" in slide and isinstance(slide["text"], str):
                            slide["content"] = [slide["text"]]
                        else:
                            slide["content"] = ["(empty)"]
                    normalized_slides.append(slide)
                presentation_outlines_json["slides"] = normalized_slides
            else:
                # as a last-ditch guarantee (should not be hit)
                presentation_outlines_json = {"slides": _heuristic_create_slides_from_text("", n_slides_to_generate)}

            # Validate into Pydantic model
            presentation_outlines = PresentationOutlineModel(**presentation_outlines_json)
            total_outlines = len(presentation_outlines.slides)

        else:
            # slides_markdown provided: each markdown block is a slide
            presentation_outlines = PresentationOutlineModel(
                slides=[SlideOutlineModel(content=slide) for slide in request.slides_markdown]
            )
            total_outlines = len(request.slides_markdown)

        # -------------------------
        # Choose layout & structure
        # -------------------------
        # Default to 'modern' layout when template not set or invalid
        template_name = (request.template or "modern").lower()
        try:
            layout_model = await get_layout_by_name(template_name)
        except Exception:
            # If requested template not found, fallback to modern (and log)
            print(f"⚠️ Layout '{template_name}' not found; falling back to 'modern'.")
            layout_model = await get_layout_by_name("modern")

        total_slide_layouts = len(layout_model.slides) if getattr(layout_model, "slides", None) else 0

        if getattr(layout_model, "ordered", False):
            presentation_structure = layout_model.to_presentation_structure()
        else:
            presentation_structure: PresentationStructureModel = await generate_presentation_structure(
                presentation_outline=presentation_outlines,
                presentation_layout=layout_model,
                instructions=request.instructions,
            )

        # Ensure structure length matches outlines
        if not isinstance(presentation_structure.slides, list):
            presentation_structure.slides = [0] * total_outlines

        # Trim or pad presentation_structure.slides to required length
        presentation_structure.slides = presentation_structure.slides[:total_outlines]
        while len(presentation_structure.slides) < total_outlines:
            presentation_structure.slides.append(random.randint(0, max(0, total_slide_layouts - 1)))

        # Sanitize slide indexes
        for idx in range(len(presentation_structure.slides)):
            if total_slide_layouts > 0 and presentation_structure.slides[idx] >= total_slide_layouts:
                presentation_structure.slides[idx] = random.randint(0, total_slide_layouts - 1)

        # -------------------------
        # Persist Presentation record
        # -------------------------
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

        # -------------------------
        # Generate slide contents & assets
        # -------------------------
        image_service = ImageGenerationService(get_images_directory())
        async_tasks: List = []
        slides: List[SlideModel] = []
        layouts = [layout_model.slides[idx] for idx in presentation_structure.slides]

        # process in batches (makes it easier to handle many slides)
        for i, layout in enumerate(layouts):
            # get slide content (LLM + formatting)
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

        # gather assets
        generated_assets_lists = await asyncio.gather(*async_tasks)
        generated_assets = [asset for assets_list in generated_assets_lists for asset in assets_list]

        # save to DB
        sql_session.add(presentation)
        sql_session.add_all(slides)
        sql_session.add_all(generated_assets)
        await sql_session.commit()

        # export
        presentation_and_path = await export_presentation(
            presentation_id, presentation.title or str(uuid.uuid4()), request.export_as
        )

        return PresentationPathAndEditPath(
            **presentation_and_path.model_dump(),
            edit_path=f"/presentation?id={presentation_id}",
        )

    except HTTPException:
        # Re-raise HTTP exceptions unchanged
        raise
    except Exception as e:
        # Improved debug output for logs
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
        # Force 'modern' template by default so front-end need not supply one
        template=kwargs.get("template", "modern"),

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

