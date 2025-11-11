# servers/fastapi/models/presentation_layout/hybrid_presenton_layout.py
from pydantic import BaseModel
from typing import List
from servers.fastapi.models.presentation_structure_model import SlideLayout
from servers.fastapi.models.presentation_layout.visual_theme_mapping import get_visual_theme

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
            # 1️⃣ Cover / Title Slide
            SlideLayout(
                id=1,
                type="title",
                style="hero-gradient",
                color_scheme=themes["hero"]["color"],
                visual=themes["hero"]["visual"],
                notes="Bold gradient hero title, startup name, and tagline with clean logo space."
            ),
            # 2️⃣ Problem Slide (General style)
            SlideLayout(
                id=2,
                type="problem",
                style="image-left-text-right",
                color_scheme=themes["problem"]["color"],
                visual="photo-office-team",
                notes="Left-side image, right-side bullet points of problems. Bold title font."
            ),
            # 3️⃣ Solution Slide (Modern)
            SlideLayout(
                id=3,
                type="solution",
                style="split-modern",
                color_scheme=themes["solution"]["color"],
                visual="product-ui-screenshot",
                notes="Clean split layout with image of solution or app UI."
            ),
            # 4️⃣ Market / Opportunity Slide
            SlideLayout(
                id=4,
                type="market",
                style="data-chart",
                color_scheme=themes["market"]["color"],
                visual="bar-chart-placeholder",
                notes="Visual chart with market data, opportunity sizing, and stats."
            ),
            # 5️⃣ Product Features (General visual cards)
            SlideLayout(
                id=5,
                type="features",
                style="image-cards",
                color_scheme=themes["features"]["color"],
                visual="product-icons-grid",
                notes="Three image cards for features or differentiators."
            ),
            # 6️⃣ Traction Slide (Modern)
            SlideLayout(
                id=6,
                type="traction",
                style="chart-focus",
                color_scheme=themes["traction"]["color"],
                visual="growth-line-chart",
                notes="Chart showing KPIs or revenue growth with clean font and icon header."
            ),
            # 7️⃣ Team Slide (General)
            SlideLayout(
                id=7,
                type="team",
                style="photo-grid",
                color_scheme=themes["team"]["color"],
                visual="avatar-cards",
                notes="Photos of team members with name and designation."
            ),
            # 8️⃣ Roadmap Slide (Modern)
            SlideLayout(
                id=8,
                type="roadmap",
                style="timeline-modern",
                color_scheme=themes["roadmap"]["color"],
                visual="timeline-icons",
                notes="Milestone-based timeline with gradient connectors."
            ),
            # 9️⃣ Financials / Graphs Slide
            SlideLayout(
                id=9,
                type="financials",
                style="chart-double",
                color_scheme=themes["financials"]["color"],
                visual="dual-chart-placeholder",
                notes="Two charts: revenue & projections with heading in bold white font."
            ),
            # 🔟 Final Call to Action
            SlideLayout(
                id=10,
                type="cta",
                style="center-cta-gradient",
                color_scheme=themes["cta"]["color"],
                visual="gradient-wave",
                notes="Final ask or thank-you slide with bold centered heading and logo."
            ),
        ]
