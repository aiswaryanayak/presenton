from typing import Optional
import uuid
from sqlalchemy import ForeignKey, Column, JSON
from sqlmodel import Field, SQLModel


class SlideModel(SQLModel, table=True):
    __tablename__ = "slides"
    __table_args__ = {"extend_existing": True}  # ✅ Prevent duplicate table registration

    id: uuid.UUID = Field(primary_key=True, default_factory=uuid.uuid4)
    presentation: uuid.UUID = Field(
        sa_column=Column(ForeignKey("presentations.id", ondelete="CASCADE"), index=True)
    )
    layout_group: str
    layout: str
    index: int
    content: dict = Field(sa_column=Column(JSON))
    html_content: Optional[str]
    speaker_note: Optional[str] = None
    properties: Optional[dict] = Field(sa_column=Column(JSON))

    def get_new_slide(self, presentation: uuid.UUID, content: Optional[dict] = None):
        return SlideModel(
            id=uuid.uuid4(),
            presentation=presentation,
            layout_group=self.layout_group,
            layout=self.layout,
            index=self.index,
            speaker_note=self.speaker_note,
            content=content or self.content,
            properties=self.properties,
        )

