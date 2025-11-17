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

# Import DB session + presentation model
from servers.fastapi.services.database import get_async_session
from servers.fastapi.models.sql.presentation import PresentationModel


async def export_presentation(
    presentation_id: uuid.UUID,
    title: str,
    export_as: Literal["pptx", "pdf"]
) -> PresentationAndPath:
    """
    Export presentation WITHOUT calling the frontend.

    Steps:
      1. Load presentation from DB
      2. Convert DB structure → PPTX model
      3. Build PPT using PptxPresentationCreator
      4. Save file to /tmp (Render-compatible)
    """

    # ---------------------------------------------
    # 1️⃣ Load Presentation From Database
    # ---------------------------------------------
    async for session in get_async_session():
        db_presentation = await session.get(PresentationModel, presentation_id)

        if not db_presentation:
            raise HTTPException(404, "Presentation not found in database")

    # ---------------------------------------------
    # 2️⃣ Convert DB structure into PPTX JSON model
    # ---------------------------------------------
    try:
        pptx_json = {
            "slides": db_presentation.structure.get("slides", []),
            "theme": db_presentation.layout,
            "title": db_presentation.title or "Untitled Presentation",
        }

        pptx_model = PptxPresentationModel.parse_obj(pptx_json)

    except Exception as e:
        raise HTTPException(
            500,
            f"❌ Failed to build PPTX model from presentation structure: {str(e)}"
        )

    # ---------------------------------------------
    # 3️⃣ Create the PPT using the Creator
    # ---------------------------------------------
    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()
    pptx_creator = PptxPresentationCreator(pptx_model, temp_dir)

    try:
        await pptx_creator.create_ppt()

    except Exception as e:
        raise HTTPException(
            500,
            f"❌ PPTX creation logic failed during slide building: {str(e)}"
        )

    # ---------------------------------------------
    # 4️⃣ Save File to /tmp directory (Render-compatible)
    # ---------------------------------------------
    try:
        export_directory = "/tmp"   # <---- FIXED LOCATION
        os.makedirs(export_directory, exist_ok=True)

        # ALWAYS save using presentation ID to match download route
        file_name = f"{presentation_id}"
        file_path = os.path.join(export_directory, f"{file_name}.pptx")

        pptx_creator.save(file_path)

    except Exception as e:
        raise HTTPException(
            500,
            f"❌ Failed to save exported PPTX file to /tmp: {str(e)}"
        )

    # ---------------------------------------------
    # 5️⃣ Return Result
    # ---------------------------------------------
    return PresentationAndPath(
        presentation_id=presentation_id,
        path=file_path
    )

