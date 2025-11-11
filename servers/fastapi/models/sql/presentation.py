from datetime import datetime
from typing import List, Optional
import uuid
from sqlalchemy import JSON, Column, DateTime, String
from sqlmodel import Boolean, Field, SQLModel

# ✅ Correct fix: import your hybrid presenton layout and alias it
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import HybridPresentonLayout as PresentationLayoutModel
from servers.fastapi.models.presentation_outline_model import PresentationOutlineModel
from servers.fastapi.models.presentation_structure_model import PresentationStructureModel
from servers.fastapi.utils.datetime_utils import get_current_utc_datetime


class PresentationModel(SQLModel, table=True):
    __tablename__ = "presentations"
    __table_args__ = {"extend_existing": True}  # ✅ Prevents duplicate table definition error

    # --- Core Fields ---
    id: uuid.UUID = Field(primary_key=True, default_factory=uuid.uuid4)
    content: str
    n_slides: int
    language: str
    title: Optional[str] = None

    # --- Metadata & Content Storage ---
    file_paths: Optional[List[str]] = Field(sa_column=Column(JSON), default=None)
    outlines: Optional[dict] = Field(sa_column=Column(JSON), default=None)
    layout: Optional[dict] = Field(sa_column=Column(JSON), default=None)
    structure: Optional[dict] = Field(sa_column=Column(JSON), default=None)

    # --- Optional Parameters ---
    instructions: Optional[str] = Field(sa_column=Column(String), default=None)
    tone: Optional[str] = Field(sa_column=Column(String), default=None)
    verbosity: Optional[str] = Field(sa_column=Column(String), default=None)
    include_table_of_contents: bool = Field(sa_column=Column(Boolean), default=False)
    include_title_slide: bool = Field(sa_column=Column(Boolean), default=True)
    web_search: bool = Field(sa_column=Column(Boolean), default=False)

    # --- Timestamps ---
    created_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            default=get_current_utc_datetime,
        ),
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            default=get_current_utc_datetime,
            onupdate=get_current_utc_datetime,
        ),
    )

    # --- Utility Methods ---

    def get_new_presentation(self):
        """Create a new PresentationModel instance duplicating current content but with a new UUID."""
        return PresentationModel(
            id=uuid.uuid4(),
            content=self.content,
            n_slides=self.n_slides,
            language=self.language,
            title=self.title,
            file_paths=self.file_paths,
            outlines=self.outlines,
            layout=self.layout,
            structure=self.structure,
            instructions=self.instructions,
            tone=self.tone,
            verbosity=self.verbosity,
            include_table_of_contents=self.include_table_of_contents,
            include_title_slide=self.include_title_slide,
            web_search=self.web_search,
        )

    def get_presentation_outline(self) -> Optional[PresentationOutlineModel]:
        """Return a PresentationOutlineModel object if outlines exist."""
        if not self.outlines:
            return None
        return PresentationOutlineModel(**self.outlines)

    def get_layout(self) -> Optional[PresentationLayoutModel]:
        """Return a PresentationLayoutModel object if layout exists."""
        if not self.layout:
            return None
        return PresentationLayoutModel(**self.layout)

    def set_layout(self, layout: PresentationLayoutModel):
        """Save layout data from a model instance."""
        self.layout = layout.model_dump()

    def get_structure(self) -> Optional[PresentationStructureModel]:
        """Return a PresentationStructureModel object if structure exists."""
        if not self.structure:
            return None
        return PresentationStructureModel(**self.structure)

    def set_structure(self, structure: PresentationStructureModel):
        """Save structure data from a model instance."""
        self.structure = structure.model_dump()
