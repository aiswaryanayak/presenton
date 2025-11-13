from typing import List, Optional, Literal
from pydantic import BaseModel


# -----------------------------
# Position & Spacing Models
# -----------------------------

class PptxPositionModel(BaseModel):
    left: int
    top: int
    width: int
    height: int

    def to_pt_list(self):
        return [self.left, self.top, self.width, self.height]

    def to_pt_xyxy(self):
        return [self.left, self.top, self.left + self.width, self.top + self.height]


class PptxSpacingModel(BaseModel):
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


# -----------------------------
# Font & Text Models
# -----------------------------

class PptxFontModel(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    color: Optional[str] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    strike: Optional[bool] = None
    font_weight: Optional[int] = None


class PptxTextRunModel(BaseModel):
    text: str
    font: Optional[PptxFontModel] = None


class PptxParagraphModel(BaseModel):
    text: Optional[str] = None
    text_runs: Optional[List[PptxTextRunModel]] = None
    alignment: Optional[str] = None
    spacing: Optional["PptxSpacingModel"] = None
    font: Optional[PptxFontModel] = None
    line_height: Optional[float] = None


# -----------------------------
# Fill & Stroke Models
# -----------------------------

class PptxFillModel(BaseModel):
    color: str = "#FFFFFF"
    opacity: float = 1.0


class PptxStrokeModel(BaseModel):
    color: str = "#000000"
    thickness: int = 0
    opacity: float = 1.0


class PptxShadowModel(BaseModel):
    color: str = "000000"
    angle: float = 45
    offset: int = 5
    radius: int = 5
    opacity: float = 0.5


# -----------------------------
# Picture / Object Fit Models
# -----------------------------

class PptxObjectFitEnum(str):
    CONTAIN = "CONTAIN"
    COVER = "COVER"
    FILL = "FILL"


class PptxObjectFitModel(BaseModel):
    fit: Optional[PptxObjectFitEnum] = None
    focus: Optional[List[float]] = None  # [x, y]


class PptxPictureModel(BaseModel):
    path: Optional[str] = None
    is_network: bool = False


# -----------------------------
# Shape Types
# -----------------------------

class PptxBoxShapeEnum(str):
    RECTANGLE = "RECTANGLE"
    CIRCLE = "CIRCLE"


class PptxAutoShapeBoxModel(BaseModel):
    type: int = 1
    position: PptxPositionModel
    paragraphs: Optional[List[PptxParagraphModel]] = None
    fill: Optional[PptxFillModel] = None
    stroke: Optional[PptxStrokeModel] = None
    shadow: Optional[PptxShadowModel] = None
    margin: Optional[PptxSpacingModel] = None
    border_radius: Optional[int] = None
    text_wrap: bool = True


class PptxPictureBoxModel(BaseModel):
    picture: PptxPictureModel
    position: PptxPositionModel
    object_fit: Optional[PptxObjectFitModel] = None
    border_radius: Optional[List[int]] = None
    shape: Optional[PptxBoxShapeEnum] = None
    opacity: Optional[float] = None
    invert: Optional[bool] = None
    clip: Optional[bool] = None
    margin: Optional[PptxSpacingModel] = None


class PptxTextBoxModel(BaseModel):
    position: PptxPositionModel
    paragraphs: List[PptxParagraphModel]
    fill: Optional[PptxFillModel] = None
    margin: Optional[PptxSpacingModel] = None
    text_wrap: bool = True


class PptxConnectorModel(BaseModel):
    type: int
    position: PptxPositionModel
    color: str = "000000"
    thickness: int = 2
    opacity: float = 1.0


# -----------------------------
# Slide + Presentation Models
# -----------------------------

class PptxSlideModel(BaseModel):
    shapes: Optional[List[
        PptxAutoShapeBoxModel
        | PptxPictureBoxModel
        | PptxTextBoxModel
        | PptxConnectorModel
    ]] = None

    background: Optional[PptxFillModel] = None
    note: Optional[str] = None


class PptxPresentationModel(BaseModel):
    slides: Optional[List[PptxSlideModel]] = None
    shapes: Optional[List[
        PptxAutoShapeBoxModel
        | PptxPictureBoxModel
        | PptxTextBoxModel
        | PptxConnectorModel
    ]] = None

