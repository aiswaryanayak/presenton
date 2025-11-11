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
    """
    Try to coerce value to an enum member of enum_cls.
    Accepts either an enum member, the enum value, or the enum name (case-insensitive).
    Falls back to default on failure.
    """
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        # Try if value is the raw enum value
        return enum_cls(value)
    except Exception:
        pass
    try:
        # Try by name (case-insensitive)
        if isinstance(value, str):
            return enum_cls[value.upper()]
    except Exception:
        pass
    return default


def _heuristic_create_slides_from_text(raw_text: str, n_slides_target: int) -> List[dict]:
    """
    Fallback heuristic: turn free-form text into `n_slides_target` slide dicts.
    We split by major separators, then if fewer segments than needed,
    split longer segments into multiple slides (by sentences).
    """
    # Normalize and remove odd markers
    t = raw_text.strip()
    # Remove triple-backticks or markdown wrappers if present
    t = re.sub(r"```(?:json)?", "", t)
    # Split by strong delimiters first: '---' or '\n\n' blocks
    segments = [s.strip() for s in re.split(r"\n-{3,}\n|\n{2,}", t) if s.strip()]

    # If nothing meaningful, use the whole text as one segment
    if not segments:
        segments = [t]

    # If too many segments, truncate
    if len(segments) >= n_slides_target:
        chosen = segments[:n_slides_target]
    else:
        # Need more slides: attempt sentence-splitting of long segments
        chosen = segments[:]
        # flatten sentences from segments until we have enough slides
        for seg in segments:
            if len(chosen) >= n_slides_target:
                break
            # split sentences (naive)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', seg) if s.strip()]
            for s in sentences:
                if len(chosen) >= n_slides_target:
                    break
                if s not in chosen:
                    chosen.append(s)
        # If still not enough, pad with the original text split into chunks
        if len(chosen) < n_slides_target:
            # split text into roughly equal slices by character length
            chunk_size = max(100, len(t) // max(1, n_slides_target))
            more = [t[i:i+chunk_size].strip() for i in range(0, len(t), chunk_size)]
            for m in more:
                if len(chosen) >= n_slides_target:
                    break
                if m and m not in chosen:
                    chosen.append(m)
        chosen = chosen[:n_slides_target]

    # Convert chosen segments to slide dictionaries
    slides = []
    for seg in chosen:
        # try to find a one-line header and rest content
        lines = [l.strip() for l in seg.splitlines() if l.strip()]
        title = lines[0] if lines else seg[:40]
        content_lines = lines[1:] if len(lines) > 1 else []
        # fallback: chunk segment into 3 bullet points if no explicit bullets
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
# Basic CRUD & Listing
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
        using_slides_markdown = False
        if request.slides_markdown:
            using_slides_markdown = True
            request.n_slides = len(request.slides_markdown)

        if not using_slides_markdown:
            additional_context = ""
            if async_status:
                async_status.message = "Generating presentation outlines"
                async_status.updated_at = datetime.now()
                sql_session.add(async_status)
                await sql_session.commit()

            if request.files:
                documents_loader = DocumentsLoader(file_paths=request.files)
                await documents_loader.load_documents()
                documents = documents_loader.documents
                if documents:
                    additional_context = "\n\n".join(documents)

            n_slides_to_generate = request.n_slides
            if request.include_table_of_contents:
                needed_toc_count = math.ceil(
                    (
                        (request.n_slides - 1)
                        if request.include_title_slide
                        else request.n_slides
                    ) / 10
                )
                n_slides_to_generate -= math.ceil(
                    (request.n_slides - needed_toc_count) / 10
                )

            presentation_outlines_text = ""
            # collect chunks from the generator (some LLM wrappers stream strings/dicts)
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
                    # append JSON dump for dict chunks
                    try:
                        presentation_outlines_text += json.dumps(chunk)
                    except Exception:
                        presentation_outlines_text += str(chunk)
                elif isinstance(chunk, str):
                    presentation_outlines_text += chunk
                else:
                    try:
                        presentation_outlines_text += json.dumps(chunk)
                    except Exception:
                        presentation_outlines_text += str(chunk)

            # Attempt to parse the collected outlines robustly
            presentation_outlines_json = None
            parse_error = None
            try:
                try:
                    presentation_outlines_json = json.loads(presentation_outlines_text)
                except Exception:
                    # try dirtyjson (tolerant)
                    presentation_outlines_json = dict(dirtyjson.loads(presentation_outlines_text))
            except Exception as e:
                parse_error = e

            # If parsing failed or slides key missing, try heuristic fallbacks
            if not isinstance(presentation_outlines_json, dict) or "slides" not in presentation_outlines_json:
                # Log debug info for server logs
                print("⚠️ DEBUG: presentation_outlines_json invalid or missing 'slides'.")
                if parse_error:
                    print("⚠️ DEBUG parse error:", repr(parse_error))
                # Try to detect a JSON array somewhere in the text like {"slides": [...]}
                try:
                    # Find first { "slides": ... } occurrence via regex (best-effort)
                    m = re.search(r'(\{.*"slides"\s*:\s*\[.*\].*?\})', presentation_outlines_text, re.DOTALL)
                    if m:
                        candidate = m.group(1)
                        presentation_outlines_json = json.loads(candidate)
                except Exception:
                    # ignore and proceed to heuristic
                    presentation_outlines_json = None

            if not isinstance(presentation_outlines_json, dict) or "slides" not in presentation_outlines_json:
                # Final fallback: construct slides heuristically from raw text
                try:
                    slides_built = _heuristic_create_slides_from_text(presentation_outlines_text, n_slides_to_generate)
                    presentation_outlines_json = {"slides": slides_built}
                    print("ℹ️ Fallback created slides from raw text (heuristic).")
                except Exception as e:
                    # Give up - return a clear HTTPException
                    traceback.print_exc()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to generate presentation outlines and fallback also failed: {str(e)}",
                    )

            # At this point, we should have a dict with "slides"
            presentation_outlines = PresentationOutlineModel(**presentation_outlines_json)
            total_outlines = len(presentation_outlines.slides)

        else:
            presentation_outlines = PresentationOutlineModel(
                slides=[SlideOutlineModel(content=slide) for slide in request.slides_markdown]
            )
            total_outlines = len(request.slides_markdown)

        # Select layout and structure
        layout_model = await get_layout_by_name(request.template)
        total_slide_layouts = len(layout_model.slides)

        if layout_model.ordered:
            presentation_structure = layout_model.to_presentation_structure()
        else:
            presentation_structure: PresentationStructureModel = (
                await generate_presentation_structure(
                    presentation_outline=presentation_outlines,
                    presentation_layout=layout_model,
                    instructions=request.instructions,
                )
            )

        # Ensure presentation_structure.slides matches total_outlines length
        presentation_structure.slides = presentation_structure.slides[:total_outlines]
        for index in range(total_outlines):
            random_slide_index = random.randint(0, total_slide_layouts - 1)
            if index >= total_outlines:
                presentation_structure.slides.append(random_slide_index)
                continue
            if presentation_structure.slides[index] >= total_slide_layouts:
                presentation_structure.slides[index] = random_slide_index

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

        image_generation_service = ImageGenerationService(get_images_directory())
        async_assets_generation_tasks = []
        slides: List[SlideModel] = []
        slide_layout_indices = presentation_structure.slides
        slide_layouts = [layout_model.slides[idx] for idx in slide_layout_indices]

        batch_size = 10
        for start in range(0, len(slide_layouts), batch_size):
            end = min(start + batch_size, len(slide_layouts))
            content_tasks = [
                get_slide_content_from_type_and_outline(
                    slide_layouts[i],
                    presentation_outlines.slides[i],
                    request.language,
                    request.tone.value,
                    request.verbosity.value,
                    request.instructions,
                )
                for i in range(start, end)
            ]
            batch_contents: List[dict] = await asyncio.gather(*content_tasks)

            for offset, slide_content in enumerate(batch_contents):
                i = start + offset
                slide_layout = slide_layouts[i]
                slide = SlideModel(
                    presentation=presentation_id,
                    layout_group=layout_model.name,
                    layout=slide_layout.id,
                    index=i,
                    speaker_note=slide_content.get("__speaker_note__"),
                    content=slide_content,
                )
                slides.append(slide)
                async_assets_generation_tasks.append(
                    process_slide_and_fetch_assets(image_generation_service, slide)
                )

        generated_assets_list = await asyncio.gather(*async_assets_generation_tasks)
        generated_assets = [asset for assets_list in generated_assets_list for asset in assets_list]

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

    except HTTPException:
        # Re-raise HTTP exceptions unchanged
        raise
    except Exception as e:
        # Improved debug output so you can check Render logs and see the real issue
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
    (presentation_id,) = (uuid.uuid4(),)
    return await generate_presentation_handler(request, presentation_id, None, sql_session)


# -------------------------
# Gemini-compatible wrapper
# -------------------------
async def generate_presentation_from_gemini(
    content: str,
    n_slides: int = 10,
    export_as: str = "pptx",
    language: str = "English",
    template: str = "modern",
    tone: str = "default",
    verbosity: str = "standard",
    instructions: str = "",
    include_title_slide: bool = True,
    include_table_of_contents: bool = False,
    web_search: bool = False,
):
    """
    Wrapper that accepts simple kwargs (as your /api/v1/ppt/from_gemini endpoint passes)
    and calls the primary internal pipeline so the output is a full Presenton-styled deck.
    """
    # Normalize enum fields robustly
    tone_enum = coerce_enum(Tone, tone, Tone.DEFAULT)
    verbosity_enum = coerce_enum(Verbosity, verbosity, Verbosity.STANDARD)

    # Build typed request
    request = GeneratePresentationRequest(
        content=content,
        n_slides=n_slides,
        export_as=export_as,
        language=language,
        template=template,
        tone=tone_enum,
        verbosity=verbosity_enum,
        instructions=instructions,
        include_title_slide=include_title_slide,
        include_table_of_contents=include_table_of_contents,
        web_search=web_search,
    )

    # Acquire a DB session and call the internal handler with a new presentation id
    async for session in get_async_session():
        presentation_id = uuid.uuid4()
        response = await generate_presentation_handler(request, presentation_id, None, session)
        return response


# -------------------------
# Backwards-compatible async entrypoint
# -------------------------
async def generate_presentation(**kwargs):
    """
    Backwards-compatible wrapper so other modules that import `generate_presentation`
    can call it using keyword args (like content=..., n_slides=..., template=..., export_as=...).
    This simply delegates to generate_presentation_from_gemini.
    """
    # Only accept expected keys; provide defaults where missing
    return await generate_presentation_from_gemini(
        content=kwargs.get("content", ""),
        n_slides=kwargs.get("n_slides", 10),
        export_as=kwargs.get("export_as", "pptx"),
        language=kwargs.get("language", "English"),
        template=kwargs.get("template", "modern"),
        tone=kwargs.get("tone", "default"),
        verbosity=kwargs.get("verbosity", "standard"),
        instructions=kwargs.get("instructions", ""),
        include_title_slide=kwargs.get("include_title_slide", True),
        include_table_of_contents=kwargs.get("include_table_of_contents", False),
        web_search=kwargs.get("web_search", False),
    )

# Keep explicit names
generate_presentation = generate_presentation
generate_presentation_from_gemini = generate_presentation_from_gemini


