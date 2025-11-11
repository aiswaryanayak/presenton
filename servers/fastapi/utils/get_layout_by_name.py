# servers/fastapi/utils/get_layout_by_name.py
from fastapi import HTTPException
from servers.fastapi.models.presentation_layout.modern import ModernEnhancedLayout

async def get_layout_by_name(layout_name: str):
    """
    Fetch the layout by name. Currently defaults to ModernEnhancedLayout
    which provides beautiful gradient, visual, and data-oriented slides.
    """
    layout_name = layout_name.lower().strip()

    # 🎨 Use ModernEnhanced layout always
    if layout_name == "modern":
        return ModernEnhancedLayout()

    # If user asks for unknown layout, still return modern
    raise HTTPException(
        status_code=404,
        detail=f"Template '{layout_name}' not found. Defaulting to modern layout."
    )
