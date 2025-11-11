# servers/fastapi/models/presentation_layout/visual_theme_mapping.py
import random

def get_visual_theme(mode: str = "hybrid"):
    """
    Returns harmonized color & visual combinations for each slide type.
    Ensures every slide looks distinct yet cohesive.
    """
    # Base palettes for hybrid startup decks
    gradients = [
        "#6C63FF,#89F7FE",
        "#FF9A9E,#FAD0C4",
        "#56CCF2,#2F80ED",
        "#A1C4FD,#C2E9FB",
        "#FDC830,#F37335"
    ]

    visuals = {
        "photo-office-team": "image of team in workspace, aesthetic bokeh",
        "product-ui-screenshot": "screenshot mockup in laptop frame",
        "bar-chart-placeholder": "3D bar chart visual",
        "growth-line-chart": "minimal line chart with gradient lines",
        "timeline-icons": "horizontal timeline with milestone icons",
        "dual-chart-placeholder": "two half-screen charts with titles",
        "avatar-cards": "grid of team portraits in white cards",
        "gradient-wave": "soft pastel gradient with abstract shapes"
    }

    def slide(c, v):
        return {"color": c, "visual": v}

    # Choose a random gradient for color consistency
    g = random.choice(gradients)

    return {
        "hero": slide(g, visuals["gradient-wave"]),
        "problem": slide(g, visuals["photo-office-team"]),
        "solution": slide(g, visuals["product-ui-screenshot"]),
        "market": slide(g, visuals["bar-chart-placeholder"]),
        "features": slide(g, visuals["product-ui-screenshot"]),
        "traction": slide(g, visuals["growth-line-chart"]),
        "team": slide(g, visuals["avatar-cards"]),
        "roadmap": slide(g, visuals["timeline-icons"]),
        "financials": slide(g, visuals["dual-chart-placeholder"]),
        "cta": slide(g, visuals["gradient-wave"]),
    }

