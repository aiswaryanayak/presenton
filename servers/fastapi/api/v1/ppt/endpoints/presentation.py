# servers/fastapi/api/v1/ppt/endpoints/presentation.py
import asyncio
import json
import random
import re
import traceback
import uuid
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional, Type

import dirtyjson
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
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

# force hybrid layout
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)

PRESENTATION_ROUTER = APIRouter(prefix="/presentation", tags=["Presentation"])


# -------------------------
# Helpers
# -------------------------
def coerce_enum(enum_cls: Type[Any], value: Any, default: Any):
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
# MISSING FUNCTION (RESTORED)
# -------------------------
async def _normalize_outlines_slides(outlines_json: dict, n_slides: int) -> dict:
    slides = outlines_json.get("slides") or []
    normalized = []
    for s in slides[:n_slides]:
        content = None
        if isinstance(s, str):
            content = s
        elif isinstance(s, dict):
            if "content" in s:
                c = s["content"]
                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    content = "\n".join(
                        [(item.get("text") if isinstance(item, dict) else str(item)) for item in c]
                    )
                else:
                    content = str(c)
            elif "text" in s:
                content = s["text"]
            else:
                vals = []
                for v in s.values():
                    if isinstance(v, (str, int, float)):
                        vals.append(str(v))
                    elif isinstance(v, list):
                        vals.extend([str(x) for x in v])
                    elif isinstance(v, dict):
                        vals.append(json.dumps(v, ensure_ascii=False))
                content = " ".join(vals) if vals else json.dumps(s, ensure_ascii=False)
        elif isinstance(s, list):
            content = "\n".join(
                [(item.get("text") if isinstance(item, dict) else str(item)) for item in s]
            )
        else:
            content = str(s)

        normalized.append({"content": content or ""})

    while len(normalized) < n_slides:
        normalized.append({"content": ""})

    return {"slides": normalized}


# -------------------------
# Structure validator (unchanged)
# -------------------------
def _validate_and_coerce_structure_for_export(structure: dict) -> dict:
    if not isinstance(structure, dict):
        raise ValueError("presentation.structure must be a dict")

    slides = structure.get("slides")
    if slides is None or not isinstance(slides, list):
        raise ValueError("presentation.structure must contain a 'slides' list")

    normalized = []
    for i, s in enumerate(slides):
        if not isinstance(s, dict):
            if isinstance(s, str):
                sdict = {"title": s, "layout_id": i + 1, "bullets": [],
                         "visuals": [], "chart_type": None, "summary": ""}
            elif isinstance(s, int):
                sdict = {"title": f"Slide {i+1}", "layout_id": int(s), "bullets": [],
                         "visuals": [], "chart_type": None, "summary": ""}
            else:
                sdict = {"title": f"Slide {i+1}", "layout_id": i + 1, "bullets": [],
                         "visuals": [], "chart_type": None, "summary": ""}
        else:
            sdict = dict(s)

        sdict["title"] = str(sdict.get("title", f"Slide {i+1}") or f"Slide {i+1}")

        try:
            sdict["layout_id"] = int(sdict.get("layout_id", i + 1) or (i + 1))
        except Exception:
            sdict["layout_id"] = i + 1

        bullets = sdict.get("bullets", [])
        if isinstance(bullets, str):
            bullets = [bullets]
        elif bullets is None:
            bullets = []
        elif not isinstance(bullets, list):
            bullets = [str(bullets)]
        bullets = [str(x) for x in bullets]
        sdict["bullets"] = bullets

        visuals = sdict.get("visuals", [])
        if isinstance(visuals, str):
            visuals = [visuals]
        elif visuals is None:
            visuals = []
        elif not isinstance(visuals, list):
            visuals = [str(visuals)]
        visuals = [str(x) for x in visuals]
        sdict["visuals"] = visuals

        sdict["chart_type"] = None if sdict.get("chart_type") is None else str(sdict.get("chart_type"))
        sdict["summary"] = "" if sdict.get("summary") is None else str(sdict.get("summary"))

        normalized.append(sdict)

    return {"slides": normalized}


# -------------------------
# MAIN GENERATION HANDLER
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
                if isinstance(chunk, (dict, list)):
                    try:
                        text_data += json.dumps(chunk, ensure_ascii=False)
                    except Exception:
                        text_data += str(chunk)
                else:
                    text_data += str(chunk)

            try:
                outlines_json = json.loads(text_data)
            except Exception:
                try:
                    outlines_json = dirtyjson.loads(text_data)
                except Exception:
                    outlines_json = {"slides": _heuristic_create_slides_from_text(text_data, n_slides)}

            if not isinstance(outlines_json, dict) or "slides" not in outlines_json:
                outlines_json = {"slides": _heur istic_create_slides_from_text(text_data, n_slides)}

            outlines_json = await _normalize_outlines_slides(outlines_json, n_slides)
            presentation_outlines = PresentationOutlineModel(**outlines_json)
            total_slides = len(presentation_outlines.slides)
        else:
            presentation_outlines = PresentationOutlineModel(
                slides=[SlideOutlineModel(content=str(slide)) for slide in request.slides_markdown]
            )
            total_slides = len(request.slides_markdown)

        layout_model = PresentationLayoutModel()

        if not hasattr(layout_model, "to_string"):
            def _layout_to_string():
                try:
                    if hasattr(layout_model, "model_dump"):
                        return json.dumps(layout_model.model_dump(), ensure_ascii=False)
                    return json.dumps(layout_model.__dict__, ensure_ascii=False)
                except Exception:
                    return str(getattr(layout_model, "__dict__", str(layout_model)))
            setattr(layout_model, "to_string", _layout_to_string)

        presentation_structure = await generate_presentation_structure(
            presentation_outline=presentation_outlines,
            presentation_layout=layout_model,
            instructions=request.instructions,
        )

        presentation = PresentationModel(
            id=presentation_id,
            content=request.content,
            n_slides=request.n_slides,
            language=request.language,
            title=get_presentation_title_from_outlines(presentation_outlines),
            outlines=presentation_outlines.model_dump(),
            layout=layout_model.model_dump() if hasattr(layout_model, "model_dump") else {},
            structure=presentation_structure.model_dump() if hasattr(presentation_structure, "model_dump") else {},
            tone=request.tone.value,
            verbosity=request.verbosity.value,
            instructions=request.instructions,
        )

        image_service = ImageGenerationService(get_images_directory())
        slides, tasks = [], []

        for i in range(total_slides):
            try:
                layout = layout_model.slides[i]
            except Exception:
                layout = layout_model.slides[0] if getattr(layout_model, "slides", None) else None

            slide_outline_text = presentation_outlines.slides[i].content if i < len(presentation_outlines.slides) else ""

            try:
                slide_type = getattr(layout, "type", "default")
                style = request.template or "hybrid-modern"
                content = await get_slide_content_from_type_and_outline(
                    slide_outline_text,
                    slide_type,
                    layout,
                    style,
                )
            except TypeError:
                content = await get_slide_content_from_type_and_outline(
                    slide_outline=slide_outline_text,
                    slide_type=slide_type,
                    layout=layout,
                    style=style,
                )

            if isinstance(content, str):
                content_dict = {"__text__": content}
                speaker_note = None
            elif isinstance(content, dict):
                content_dict = content
                speaker_note = content.get("__speaker_note__")
            else:
                content_dict = {"__raw__": str(content)}
                speaker_note = None

            slide = SlideModel(
                presentation=presentation_id,
                layout_group=getattr(layout_model, "name", "hybrid_presenton"),
                layout=getattr(layout, "id", i),
                index=i,
                speaker_note=speaker_note,
                content=content_dict,
            )
            slides.append(slide)
            tasks.append(process_slide_and_fetch_assets(image_service, slide))

        generated_assets = [asset for group in await asyncio.gather(*tasks) for asset in group]

        sql_session.add(presentation)
        sql_session.add_all(slides)
        sql_session.add_all(generated_assets)
        await sql_session.commit()

        try:
            debug_structure = presentation.structure
        except Exception:
            debug_structure = {}
        print("🧩 DEBUG Presentation Structure:", json.dumps(debug_structure, ensure_ascii=False))

        try:
            coerced_structure = _validate_and_coerce_structure_for_export(
                presentation.structure
            )
        except Exception as ex:
            print("🔥 ERROR: structure validation failed:", repr(ex))
            raise HTTPException(status_code=500, detail=f"Presentation structure invalid: {str(ex)}")

        presentation.structure = coerced_structure

        try:
            presentation_path = await export_presentation(
                presentation_id,
                presentation.title or str(uuid.uuid4()),
                request.export_as,
            )
        except Exception as ex:
            print("🔥 ERROR: export_presentation failed:", repr(ex))
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"failed to convert to pptx: {str(ex)}")

        return PresentationPathAndEditPath(
            **presentation_path.model_dump(),
            edit_path=f"/presentation?id={presentation_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 DEBUG: Presentation generation failed:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate presentation: {str(e)}")


# -------------------------
# Public sync endpoint
# -------------------------
@PRESENTATION_ROUTER.post("/generate", response_model=PresentationPathAndEditPath)
async def generate_presentation_sync(
    request: GeneratePresentationRequest,
    sql_session: AsyncSession = Depends(get_async_session),
):
    return await generate_presentation_handler(request, uuid.uuid4(), sql_session)


# -------------------------
# FIXED DOWNLOAD ENDPOINT
# -------------------------
@PRESENTATION_ROUTER.get("/download/{presentation_id}")
async def download_presentation(presentation_id: uuid.UUID):
    """
    FIX: Always load PPTX from /tmp because Render only persists files there.
    """
    file_path = f"/tmp/{presentation_id}.pptx"

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"No exported PPTX found for presentation {presentation_id}"
        )

    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=f"{presentation_id}.pptx"
    )


# -------------------------
# Gemini wrapper
# -------------------------
async def generate_presentation_from_gemini(**kwargs):
    tone_enum = coerce_enum(Tone, kwargs.get("tone", "default"), Tone.DEFAULT)
    verbosity_enum = coerce_enum(Verbosity, kwargs.get("verbosity", "standard"), Verbosity.STANDARD)

    request = GeneratePresentationRequest(
        content=kwargs.get("content", ""),
        n_slides=kwargs.get("n_slides", 10),
        export_as=kwargs.get("export_as", "pptx"),
        language=kwargs.get("language", "English"),
        template="hybrid",
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
