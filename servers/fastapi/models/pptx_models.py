# servers/fastapi/models/pptx_models.py
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field


# -------------------------
# Enums
# -------------------------
class PptxBoxShapeEnum(int, Enum):
    RECTANGLE = 1
    ROUND_RECTANGLE = 2
    ELLIPSE = 3
    CIRCLE = 4
    # add more if needed


class PptxObjectFitEnum(str, Enum):
    CONTAIN = "contain"
    COVER = "cover"
    FILL = "fill"


# -------------------------
# Small helper models
# -------------------------
class PptxFontModel(BaseModel):
    name: Optional[str] = None
    size: Optional[float] = None
    color: Optional[str] = None
    font_weight: Optional[int] = None
    italic: Optional[bool] = False
    underline: Optional[bool] = False
    strike: Optional[bool] = False

    model_config = {"extra": "allow"}


class PptxTextRunModel(BaseModel):
    text: str = ""
    font: Optional[PptxFontModel] = None

    model_config = {"extra": "allow"}


class PptxParagraphModel(BaseModel):
    text: Optional[str] = None
    text_runs: Optional[List[PptxTextRunModel]] = None
    spacing: Optional[dict] = None
    line_height: Optional[float] = None
    alignment: Optional[str] = None
    font: Optional[PptxFontModel] = None

    model_config = {"extra": "allow"}


class PptxPositionModel(BaseModel):
    left: int = 0
    top: int = 0
    width: int = 100
    height: int = 100

    def to_pt_list(self):
        # your service expects this method
        return [self.left, self.top, self.width, self.height]

    def to_pt_xyxy(self):
        # approximate for connector add
        return [self.left, self.top, self.left + self.width, self.top + self.height]


class PptxSpacingModel(BaseModel):
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


# -------------------------
# Object-Fit Model used by image utils
# -------------------------
class PptxObjectFitModel(BaseModel):
    fit: Optional[PptxObjectFitEnum] = None
    focus: Optional[List[float]] = None  # [x_percent, y_percent]

    model_config = {"extra": "allow"}


# -------------------------
# Visuals / Fill / Stroke / Shadow
# -------------------------
class PptxFillModel(BaseModel):
    color: Optional[str] = "#FFFFFF"
    opacity: Optional[float] = 1.0


class PptxStrokeModel(BaseModel):
    color: Optional[str] = "#000000"
    thickness: Optional[float] = 1.0
    opacity: Optional[float] = 1.0


class PptxShadowModel(BaseModel):
    color: str = "000000"
    radius: float = 4.0
    offset: float = 2.0
    angle: float = 0.0
    opacity: float = 0.3


# -------------------------
# Picture, Textbox, Autoshape, Connector models
# -------------------------
class PptxPictureRefModel(BaseModel):
    path: Optional[str] = None
    is_network: Optional[bool] = False


class PptxPictureBoxModel(BaseModel):
    picture: PptxPictureRefModel
    position: PptxPositionModel
    margin: Optional[PptxSpacingModel] = None
    border_radius: Optional[Union[List[int], int]] = None
    object_fit: Optional[PptxObjectFitModel] = None
    clip: Optional[bool] = False
    shape: Optional[PptxBoxShapeEnum] = None
    invert: Optional[bool] = False
    opacity: Optional[float] = None


class PptxTextBoxModel(BaseModel):
    position: PptxPositionModel
    paragraphs: Optional[List[PptxParagraphModel]] = []
    text_wrap: Optional[bool] = True
    margin: Optional[PptxSpacingModel] = None
    fill: Optional[PptxFillModel] = None


class PptxAutoShapeBoxModel(BaseModel):
    type: int
    position: PptxPositionModel
    paragraphs: Optional[List[PptxParagraphModel]] = []
    text_wrap: Optional[bool] = True
    fill: Optional[PptxFillModel] = None
    stroke: Optional[PptxStrokeModel] = None
    shadow: Optional[PptxShadowModel] = None
    border_radius: Optional[Union[int, List[int]]] = None


class PptxConnectorModel(BaseModel):
    type: int
    position: PptxPositionModel
    thickness: float = 1.0
    color: str = "000000"
    opacity: Optional[float] = 1.0


# -------------------------
# Slide / Presentation model (simplified)
# -------------------------
class PptxSlideModel(BaseModel):
    index: Optional[int] = 0
    note: Optional[str] = None
    shapes: Optional[List[Union[PptxPictureBoxModel, PptxTextBoxModel, PptxAutoShapeBoxModel, PptxConnectorModel]]] = []
    background: Optional[PptxFillModel] = None


class PptxPresentationModel(BaseModel):
    title: Optional[str] = None
    slides: Optional[List[PptxSlideModel]] = []
    shapes: Optional[List[Union[PptxPictureBoxModel, PptxAutoShapeBoxModel, PptxTextBoxModel]]] = []

    model_config = {"extra": "allow"}
