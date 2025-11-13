from html.parser import HTMLParser
from typing import List, Optional

# IMPORTANT:
# We ONLY import the dataclasses used.
# We DO NOT import anything that imports this file back.
from models.pptx_models import PptxFontModel, PptxTextRunModel


class InlineHTMLToRunsParser(HTMLParser):
    """
    Converts inline HTML (b, i, u, br, code, strong, etc.)
    into a list of PptxTextRunModel objects with font styling applied.
    """

    def __init__(self, base_font: PptxFontModel):
        super().__init__(convert_charrefs=True)
        self.base_font = base_font or PptxFontModel()
        self.tag_stack: List[str] = []
        self.text_runs: List[PptxTextRunModel] = []

    # -------------------------------
    # Determine current font from tag stack
    # -------------------------------
    def _current_font(self) -> PptxFontModel:
        font_dict = self.base_font.model_dump()

        is_bold = any(tag in ("strong", "b") for tag in self.tag_stack)
        is_italic = any(tag in ("i", "em") for tag in self.tag_stack)
        is_underline = any(tag == "u" for tag in self.tag_stack)
        is_strike = any(tag in ("s", "strike", "del") for tag in self.tag_stack)
        is_code = any(tag == "code" for tag in self.tag_stack)

        if is_bold:
            font_dict["font_weight"] = 700
        if is_italic:
            font_dict["italic"] = True
        if is_underline:
            font_dict["underline"] = True
        if is_strike:
            font_dict["strike"] = True
        if is_code:
            font_dict["name"] = "Courier New"

        return PptxFontModel(**font_dict)

    # -------------------------------
    # Tag handlers
    # -------------------------------
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        # line break
        if tag == "br":
            self.text_runs.append(PptxTextRunModel(text="\n"))
            return

        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        # remove only the last matching tag
        for i in range(len(self.tag_stack) - 1, -1, -1):
            if self.tag_stack[i] == tag:
                del self.tag_stack[i]
                break

    def handle_data(self, data):
        if not data:
            return
        run_font = self._current_font()
        self.text_runs.append(PptxTextRunModel(text=data, font=run_font))


# -------------------------------------------------
# FINAL PUBLIC API (called by pptx_presentation_creator)
# -------------------------------------------------
def parse_html_text_to_text_runs(
    text: str,
    base_font: Optional[PptxFontModel] = None
) -> List[PptxTextRunModel]:
    """
    Convert inline HTML into text runs with styling.
    Safe, no circular imports, correct call signature.
    """

    if not text:
        return []

    normalized = (
        text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\n", "<br>")
    )

    parser = InlineHTMLToRunsParser(base_font if base_font else PptxFontModel())
    parser.feed(normalized)

    return parser.text_runs


