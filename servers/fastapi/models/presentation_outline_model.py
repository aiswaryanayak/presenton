from typing import List, Union
from pydantic import BaseModel


class SlideOutlineModel(BaseModel):
    # ✅ Allow both plain text and list of bullet points
    content: Union[str, List[str]]

    def get_text(self) -> str:
        """Convert list content to a clean text string if needed."""
        if isinstance(self.content, list):
            # Join bullets nicely if it's a list
            return "\n".join(f"• {item}" for item in self.content)
        return str(self.content or "")


class PresentationOutlineModel(BaseModel):
    slides: List[SlideOutlineModel]

    def to_string(self) -> str:
        """Return the slides as readable markdown-style text."""
        message = ""
        for i, slide in enumerate(self.slides):
            message += f"## Slide {i+1}:\n"
            message += f"{slide.get_text()}\n\n"
        return message.strip()
