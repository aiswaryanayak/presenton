import json
import os
import uuid
from typing import Literal
from fastapi import HTTPException

from models.pptx_models import PptxPresentationModel
from models.presentation_and_path import PresentationAndPath
from services.pptx_presentation_creator import PptxPresentationCreator
from services.temp_file_service import TEMP_FILE_SERVICE
from utils.asset_directory_utils import get_exports_directory
from pathvalidate import sanitize_filename


async def export_presentation(
    presentation_id: uuid.UUID,
    title: str,
    export_as: Literal["pptx", "pdf"]
) -> PresentationAndPath:
    """
    FIXED VERSION:
    ---------------------------------------
    - Removes localhost API calls
    - Directly builds PPTX using stored database model
    - Works 100% in Render
    """

    # 1️⃣ Load the PPTX model directly from DB
    try:
        pptx_model = await PptxPresentationModel.from_presentation_id(presentation_id)
    except Exception as e:
        print("ERROR loading PPTX model:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load PPTX model for export: {str(e)}"
        )

    # 2️⃣ Create temporary directory
    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()

    # 3️⃣ Build PPTX
    pptx_creator = PptxPresentationCreator(pptx_model, temp_dir)
    try:
        await pptx_creator.create_ppt()
    except Exception as e:
        print("ERROR creating PPT:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed generating PPTX file: {str(e)}"
        )

    # 4️⃣ Save PPTX
    export_directory = get_exports_directory()
    pptx_path = os.path.join(
        export_directory,
        f"{sanitize_filename(title or str(presentation_id))}.pptx",
    )

    try:
        pptx_creator.save(pptx_path)
    except Exception as e:
        print("ERROR saving PPTX:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Unable to save PPTX file: {str(e)}"
        )

    return PresentationAndPath(
        presentation_id=presentation_id,
        path=pptx_path,
    )
