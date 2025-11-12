# servers/fastapi/models/presentation_outline_model.py

from typing import List, Union
from pydantic import BaseModel
import json


class SlideOutlineModel(BaseModel):
    """
    Represents a single slide's outline content.
    Can be a plain string or a list of bullet points.
    """
    content: Union[str, List[str]]

    def get_text(self) -> str:
        """Convert list or structured content to a readable text string."""
        try:
            if isinstance(self.content, list):
                # Join list elements with bullets
                return "\n".join(f"• {str(item)}" for item in self.content)
            elif isinstance(self.content, dict):
                # Flatten dict content into key-value pairs
                return "\n".join(f"{k}: {v}" for k, v in self.content.items())
            return str(self.content or "")
        except Exception:
            # Fallback for non-stringifiable data
            return json.dumps(self.content, ensure_ascii=False)


class PresentationOutlineModel(BaseModel):
    """
    Represents a full presentation outline consisting of multiple slides.
    """
    slides: List[SlideOutlineModel]

    def to_string(self) -> str:
        """
        Convert all slides into a human-readable markdown string for LLM input.
        Prevents type errors from lists, dicts, or nested structures.
        """
        text_parts = []
        for i, slide in enumerate(self.slides):
            try:
                content_text = slide.get_text()
            except Exception:
                # In case get_text fails for weird content
                content_text = json.dumps(getattr(slide, "content", ""), ensure_ascii=False)

            text_parts.append(f"## Slide {i+1}\n{content_text}")

        return "\n\n".join(text_parts).strip()
