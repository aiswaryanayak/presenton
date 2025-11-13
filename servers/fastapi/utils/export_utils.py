# servers/fastapi/utils/export_utils.py
import os
import uuid
import json
import asyncio
from typing import Literal, Optional

import aiohttp
from aiohttp import ClientTimeout, ClientError
from fastapi import HTTPException

from models.pptx_models import PptxPresentationModel
from models.presentation_and_path import PresentationAndPath
from services.pptx_presentation_creator import PptxPresentationCreator
from services.temp_file_service import TEMP_FILE_SERVICE
from utils.asset_directory_utils import get_exports_directory
from pathvalidate import sanitize_filename

# Default frontend endpoint (change if you host it somewhere else).
# We prefer an environment variable so you can override in production.
DEFAULT_FRONTEND_BASE = os.environ.get(
    "PRESENTON_FRONTEND_URL", "https://ai-fundraising-support.vercel.app"
).rstrip("/")

# The exact path Next.js exposes
PPTX_MODEL_ROUTE = os.environ.get(
    "PRESENTON_PPTX_MODEL_ROUTE", "/api/presentation_to_pptx_model"
)

# Full URL builder
def _build_presentation_model_url(presentation_id: uuid.UUID) -> str:
    base = DEFAULT_FRONTEND_BASE
    return f"{base}{PPTX_MODEL_ROUTE}?id={presentation_id}"


# Network parameters
REQUEST_TIMEOUT_SEC = int(os.environ.get("EXPORT_HTTP_TIMEOUT_SEC", "20"))
REQUEST_RETRIES = int(os.environ.get("EXPORT_HTTP_RETRIES", "2"))
RETRY_DELAY_SEC = float(os.environ.get("EXPORT_HTTP_RETRY_DELAY", "1.0"))


async def _fetch_pptx_model_json(presentation_id: uuid.UUID) -> dict:
    """
    Fetch the PPTX model JSON from the Next.js endpoint.
    Retries on transient errors.
    """
    url = _build_presentation_model_url(presentation_id)
    timeout = ClientTimeout(total=REQUEST_TIMEOUT_SEC)
    last_exc: Optional[Exception] = None

    for attempt in range(1, REQUEST_RETRIES + 2):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"Accept": "application/json"}) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Failed to fetch PPTX model (status={resp.status}): {text[:400]}",
                        )
                    try:
                        data = json.loads(text)
                    except Exception as e:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Failed to parse JSON from PPTX model endpoint: {str(e)} - raw: {text[:400]}",
                        )
                    return data
        except (ClientError, asyncio.TimeoutError) as e:
            last_exc = e
            # transient error -> retry with small backoff
            if attempt <= REQUEST_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SEC * attempt)
                continue
            # exhausted retries -> raise
            raise HTTPException(
                status_code=502,
                detail=f"Network error fetching PPTX model from frontend: {str(last_exc)}",
            )
        except HTTPException:
            # rethrow HTTPExceptions from inside
            raise
        except Exception as e:
            # unexpected error -> wrap and raise
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error fetching PPTX model: {str(e)}",
            )

    # unreachable
    raise HTTPException(status_code=502, detail="Failed to fetch PPTX model")


def _validate_basic_pptx_json(data: dict):
    """
    Perform light validation on the PPTX JSON before passing into Pydantic.
    This protects against obviously malformed responses.
    """
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="PPTX model response is not an object")

    # slides is required and must be a list
    slides = data.get("slides")
    if slides is None or not isinstance(slides, list):
        raise HTTPException(status_code=502, detail="PPTX model missing 'slides' list")

    if len(slides) == 0:
        # allow zero slides but warn (return value might still be valid)
        # we'll still allow converting to the model; not an immediate error
        pass


async def export_presentation(
    presentation_id: uuid.UUID,
    title: str,
    export_as: Literal["pptx", "pdf"]
) -> PresentationAndPath:
    """
    Export presentation by:
      1. Fetching PPTX JSON model from Next.js frontend
      2. Validating/parsing into PptxPresentationModel
      3. Creating PPTX with PptxPresentationCreator
      4. Saving file to exports directory and returning path
    """
    # 1. Fetch PPTX JSON from frontend
    try:
        pptx_json = await _fetch_pptx_model_json(presentation_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve PPTX model: {str(e)}")

    # 2. Basic validation
    try:
        _validate_basic_pptx_json(pptx_json)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"PPTX model validation error: {str(e)}")

    # 3. Parse into PptxPresentationModel (pydantic)
    try:
        # if your model uses a different constructor, adapt here.
        pptx_model = PptxPresentationModel.parse_obj(pptx_json)
    except Exception as e:
        # emit the JSON shape in logs for debugging (trimmed)
        excerpt = json.dumps(pptx_json)[:1000]
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse PPTX model into PptxPresentationModel: {str(e)} - json_excerpt={excerpt}"
        )

    # 4. Create temp dir and generate PPTX (creator expects a model + temp dir)
    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()
    pptx_creator = PptxPresentationCreator(pptx_model, temp_dir)

    try:
        # create_ppt is async in your earlier code; await it
        await pptx_creator.create_ppt()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPTX creation failed: {str(e)}")

    # 5. Save PPTX file
    try:
        export_directory = get_exports_directory()
        os.makedirs(export_directory, exist_ok=True)
        file_name = sanitize_filename(title or str(presentation_id))
        file_path = os.path.join(export_directory, f"{file_name}.pptx")
        pptx_creator.save(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PPTX file: {str(e)}")

    # 6. Return result
    return PresentationAndPath(presentation_id=presentation_id, path=file_path)

