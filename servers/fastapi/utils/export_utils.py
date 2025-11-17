# servers/fastapi/utils/export_utils.py

import os
import uuid
import json
from typing import Literal
from fastapi import HTTPException

from models.pptx_models import PptxPresentationModel
from models.presentation_and_path import PresentationAndPath
from services.pptx_presentation_creator import PptxPresentationCreator
from services.temp_file_service import TEMP_FILE_SERVICE
from utils.asset_directory_utils import get_exports_directory
from pathvalidate import sanitize_filename

# DB
from servers.fastapi.services.database import get_async_session
from servers.fastapi.models.sql.presentation import PresentationModel


async def export_presentation(
    presentation_id: uuid.UUID,
    title: str,
    export_as: Literal["pptx", "pdf"]
) -> PresentationAndPath:
    """
    EXPORTS PRESENTATION USING ONLY UUID AS FINAL FILE NAME.
    This guarantees the download endpoint ALWAYS finds the file.
    """

    # ---------------------------
    # 1️⃣ Load DB Presentation
    # ---------------------------
    async for session in get_async_session():
        db_presentation = await session.get(PresentationModel, presentation_id)
        if not db_presentation:
            raise HTTPException(404, "Presentation not found")

    # ---------------------------
    # 2️⃣ Build PPTX Model
    # ---------------------------
    try:
        pptx_json = {
            "slides": db_presentation.structure.get("slides", []),
            "theme": db_presentation.layout,
            "title": db_presentation.title or "Untitled"
        }
        pptx_model = PptxPresentationModel.parse_obj(pptx_json)

    except Exception as e:
        raise HTTPException(
            500, f"Failed to prepare PPTX model: {str(e)}"
        )

    # ---------------------------
    # 3️⃣ Create PPT in Temp Dir
    # ---------------------------
    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()
    pptx_creator = PptxPresentationCreator(pptx_model, temp_dir)

    try:
        await pptx_creator.create_ppt()
    except Exception as e:
        raise HTTPException(500, f"PPTX creation failed: {str(e)}")

    # ---------------------------
    # 4️⃣ Save to Exports Directory
    # FINAL FILE NAME → ALWAYS UUID
    # ---------------------------
    try:
        export_dir = get_exports_directory()
        os.makedirs(export_dir, exist_ok=True)

        # FINAL FIX — ALWAYS SAVE USING UUID
        final_filename = f"{presentation_id}.pptx"
        final_path = os.path.join(export_dir, final_filename)

        pptx_creator.save(final_path)

    except Exception as e:
        raise HTTPException(
            500, f"Failed to save exported file: {str(e)}"
        )

    # ---------------------------
    # 5️⃣ Return Path For Frontend
    # ---------------------------
    return PresentationAndPath(
        presentation_id=presentation_id,
        path=final_path
    )
