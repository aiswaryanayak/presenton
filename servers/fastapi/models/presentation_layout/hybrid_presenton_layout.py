# servers/fastapi/models/presentation_layout/hybrid_presenton_layout.py

from pydantic import BaseModel
from typing import List, Optional
from servers.fastapi.models.presentation_layout.visual_theme_mapping import get_visual_theme


class SlideLayout(BaseModel):
    """
    Represents a single slide layout type for the Hybrid Presenton theme.
    Each layout controls visual, color, and content structure.
    """
    id: int
    type: str
    style: str
    color_scheme: str
    visual: Optional[str] = None
    notes: Optional[str] = None


class HybridPresentonLayout(BaseModel):
    """
    Combines Presenton's Modern + General templates for beautiful AI decks.
    Features: image-rich layouts, data charts, gradients, and bold text.
    """
    name: str = "hybrid_presenton"
    ordered: bool = True
    slides: List[SlideLayout] = []

    def __init__(self, **data):
        super().__init__(**data)
        themes = get_visual_theme("hybrid")

        self.slides = [
            SlideLayout(
                id=1,
                type="title",
                style="hero-gradient",
                color_scheme=themes.get("hero", {}).get("color", "#6C63FF"),
                visual=themes.get("hero", {}).get("visual", "gradient-bg"),
                notes="Bold gradient hero title with startup name, tagline, and logo space."
            ),
            SlideLayout(
                id=2,
                type="problem",
                style="image-left-text-right",
                color_scheme=themes.get("problem", {}).get("color", "#F5F7FB"),
                visual="photo-office-team",
                notes="Left-side image, right-side bullet points of problems with bold headings."
            ),
            SlideLayout(
                id=3,
                type="solution",
                style="split-modern",
                color_scheme=themes.get("solution", {}).get("color", "#EEF1FF"),
                visual="product-ui-screenshot",
                notes="Split layout with product visuals or screenshots and clean typography."
            ),
            SlideLayout(
                id=4,
                type="market",
                style="data-chart",
                color_scheme=themes.get("market", {}).get("color", "#E9F5FF"),
                visual="bar-chart-placeholder",
                notes="Market size chart with opportunity highlights and metrics."
            ),
            SlideLayout(
                id=5,
                type="features",
                style="image-cards",
                color_scheme=themes.get("features", {}).get("color", "#F9F9FF"),
                visual="icons-grid",
                notes="Feature cards with icons and small descriptions."
            ),
            SlideLayout(
                id=6,
                type="traction",
                style="chart-focus",
                color_scheme=themes.get("traction", {}).get("color", "#DDF2FF"),
                visual="growth-line-chart",
                notes="Clean chart layout for growth metrics and KPIs."
            ),
            SlideLayout(
                id=7,
                type="team",
                style="photo-grid",
                color_scheme=themes.get("team", {}).get("color", "#FFFFFF"),
                visual="avatar-cards",
                notes="Photos of team members with roles and social icons."
            ),
            SlideLayout(
                id=8,
                type="roadmap",
                style="timeline-modern",
                color_scheme=themes.get("roadmap", {}).get("color", "#F2F8FF"),
                visual="timeline-icons",
                notes="Milestone-based roadmap with gradient timeline."
            ),
            SlideLayout(
                id=9,
                type="financials",
                style="chart-double",
                color_scheme=themes.get("financials", {}).get("color", "#E6F0FF"),
                visual="dual-chart-placeholder",
                notes="Two charts: revenue & projections with bold white heading."
            ),
            SlideLayout(
                id=10,
                type="cta",
                style="center-cta-gradient",
                color_scheme=themes.get("cta", {}).get("color", "#6C63FF"),
                visual="gradient-wave",
                notes="Final call-to-action slide with gradient background and centered text."
            ),
            SlideLayout(
                id=11,
                type="quote",
                style="highlight-bold",
                color_scheme=themes.get("quote", {}).get("color", "#F4F4F6"),
                visual="minimal-abstract",
                notes="Single bold quote or tagline centered with visual emphasis."
            ),
            SlideLayout(
                id=12,
                type="comparison",
                style="two-column",
                color_scheme=themes.get("comparison", {}).get("color", "#FFFFFF"),
                visual="before-after-illustration",
                notes="Side-by-side comparison layout for transformation visuals."
            ),
        ]

    def to_string(self) -> str:
        """
        Convert this layout into a readable summary string for LLM context.
        Helps Gemini understand available slide types and visual patterns.
        """
        summary = "# HYBRID PRESENTON LAYOUT OVERVIEW\n"
        summary += f"Theme: {self.name}\n"
        summary += "Each layout combines modern gradient visuals and clean data presentation.\n\n"

        for slide in self.slides:
            summary += (
                f"- [{slide.id}] {slide.type.upper()} | Style: {slide.style} | "
                f"Color: {slide.color_scheme} | Visual: {slide.visual or 'N/A'}\n"
                f"  → {slide.notes}\n\n"
            )

        return summary.strip()



