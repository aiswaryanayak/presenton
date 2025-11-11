# servers/fastapi/utils/get_layout_by_name.py

from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import HybridPresentonLayout

async def get_layout_by_name(layout_name: str):
    """
    Fetches the layout by name.
    Currently defaults to the Hybrid Presenton Layout that combines
    Modern and General templates for rich, visual slides.
    """
    layout_name = layout_name.lower().strip()

    # 🎨 Supported layouts
    if layout_name in ["modern", "general", "hybrid", "default", "presenton"]:
        return HybridPresentonLayout()

    # 🔁 Gracefully fall back to hybrid
    return HybridPresentonLayout()
