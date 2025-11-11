# servers/fastapi/models/sql/template.py

from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field, Column, JSON, String
from sqlalchemy import DateTime
from servers.fastapi.utils.datetime_utils import get_current_utc_datetime


class TemplateModel(SQLModel, table=True):
    """
    Represents a reusable presentation template (layout + metadata).
    Stores template name, layout structure, and styling data for hybrid/modern decks.
    """

    __tablename__ = "templates"
    __table_args__ = {"extend_existing": True}

    # ✅ Fixed: remove index=True when using sa_column
    id: uuid.UUID = Field(primary_key=True, default_factory=uuid.uuid4)
    name: str = Field(sa_column=Column(String, nullable=False, unique=True))
    description: str = Field(default="", sa_column=Column(String, nullable=True))
    layout_data: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    theme: str = Field(default="hybrid", sa_column=Column(String, nullable=False))

    created_at: datetime = Field(
        default_factory=get_current_utc_datetime,
        sa_column=Column(DateTime(timezone=True), default=get_current_utc_datetime)
    )
    updated_at: datetime = Field(
        default_factory=get_current_utc_datetime,
        sa_column=Column(DateTime(timezone=True), onupdate=get_current_utc_datetime)
    )

    def summary(self) -> dict:
        """Return a short summary of this template."""
        return {
            "id": str(self.id),
            "name": self.name,
            "theme": self.theme,
            "description": self.description or "No description provided.",
        }

    def export_for_frontend(self) -> dict:
        """Prepare lightweight structure for API/frontend use."""
        return {
            "name": self.name,
            "theme": self.theme,
            "layout_data": self.layout_data,
            "updated_at": self.updated_at.isoformat(),
        }
