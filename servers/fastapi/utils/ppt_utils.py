# servers/fastapi/utils/ppt_utils.py

import re
from typing import List
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)
from servers.fastapi.models.presentation_outline_model import PresentationOutlineModel
from servers.fastapi.models.presentation_structure_model import PresentationStructureModel


def get_presentation_title_from_outlines(
    presentation_outlines: PresentationOutlineModel,
) -> str:
    """Extract a clean, human-readable title from the first slide content."""
    if not presentation_outlines.slides:
        return "Untitled Presentation"

    first_content = presentation_outlines.slides[0].content or ""

    # Remove headings like “# Page 1” or similar markdown titles
    if re.match(r"^\s*#{1,6}\s*Page\s+\d+\b", first_content):
        first_content = re.sub(
            r"^\s*#{1,6}\s*Page\s+\d+\b[\s,:\-]*",
            "",
            first_content,
            count=1,
        )

    # Clean formatting symbols and return a short title
    return (
        first_content[:100]
        .replace("#", "")
        .replace("/", "")
        .replace("\\", "")
        .replace("\n", " ")
        .strip()
        or "Untitled Presentation"
    )


def find_slide_layout_index_by_regex(
    layout: PresentationLayoutModel, patterns: List[str]
) -> int:
    """Find the slide layout index that matches any of the given regex patterns."""
    def _find_index(pattern: str) -> int:
        regex = re.compile(pattern, re.IGNORECASE)
        for index, slide_layout in enumerate(layout.slides):
            candidates = [
                getattr(slide_layout, "id", ""),
                getattr(slide_layout, "type", ""),
                getattr(slide_layout, "style", ""),
                getattr(slide_layout, "notes", ""),
            ]
            for text in candidates:
                if text and regex.search(str(text)):
                    return index
        return -1

    for pattern in patterns:
        match_index = _find_index(pattern)
        if match_index != -1:
            return match_index

    return -1


def select_toc_or_list_slide_layout_index(
    layout: PresentationLayoutModel,
) -> int:
    """Pick the best layout index for Table of Contents or List slides."""
    toc_patterns = [
        r"\btable\s*of\s*contents\b",
        r"\bagenda\b",
        r"\bcontents\b",
        r"\boutline\b",
        r"\bindex\b",
        r"\btoc\b",
    ]

    list_patterns = [
        r"\b(bullet(ed)?\s*list|bullets?)\b",
        r"\b(numbered\s*list|ordered\s*list|unordered\s*list)\b",
        r"\blist\b",
    ]

    toc_index = find_slide_layout_index_by_regex(layout, toc_patterns)
    if toc_index != -1:
        return toc_index

    return find_slide_layout_index_by_regex(layout, list_patterns)

