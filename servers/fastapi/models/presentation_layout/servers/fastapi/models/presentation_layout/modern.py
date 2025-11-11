# servers/fastapi/models/presentation_layout/modern.py
from pydantic import BaseModel
from typing import List
from servers.fastapi.models.presentation_structure_model import SlideLayout

class ModernEnhancedLayout(BaseModel):
    """
    Modern+Enhanced layout for beautiful, non-generic Presenton decks.
    Balanced visuals, charts, and color-coded sections.
    """
    name: str = "modern"
    ordered: bool = False
    slides: List[SlideLayout] = [
        # 1️⃣ Hero / Cover
        SlideLayout(
            id=1,
            type="title",
            style="hero-gradient",
            color_scheme="purple-pink",
            visual="abstract-gradient-wave",
            notes="Big bold title with gradient background and subtle AI-illustration."
        ),
        # 2️⃣ Vision / Problem Statement
        SlideLayout(
            id=2,
            type="content",
            style="split-vision",
            color_scheme="teal-blue",
            visual="minimal-illustration",
            notes="Left-aligned text, right-side visual; use short bold phrases."
        ),
        # 3️⃣ Solution / Product Slide
        SlideLayout(
            id=3,
            type="visual",
            style="image-focus",
            color_scheme="violet",
            visual="product-ui-mockup",
            notes="Use large product visuals or mockups; minimal text."
        ),
        # 4️⃣ Data / Insights Slide
        SlideLayout(
            id=4,
            type="data",
            style="chart-focus",
            color_scheme="blue-green",
            visual="chart-placeholder",
            notes="Includes space for metrics or growth graph."
        ),
        # 5️⃣ Team Slide
        SlideLayout(
            id=5,
            type="team",
            style="portrait-cards",
            color_scheme="orange-coral",
            visual="avatars",
            notes="Show team members with roles; clean card layout."
        ),
        # 6️⃣ Roadmap / Timeline
        SlideLayout(
            id=6,
            type="timeline",
            style="modern-line",
            color_scheme="indigo",
            visual="timeline-graphic",
            notes="Use icons or milestones across a horizontal timeline."
        ),
        # 7️⃣ Call-to-Action / Summary
        SlideLayout(
            id=7,
            type="summary",
            style="center-highlight",
            color_scheme="gradient-violet",
            visual="abstract-end-art",
            notes="Inspirational closing slide with strong CTA (e.g. 'Join us', 'Let’s build the future')."
        ),
    ]


