# servers/fastapi/utils/get_layout_by_name.py
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import HybridPresentonLayout

async def get_layout_by_name(layout_name: str):
    """
    Returns the enhanced hybrid layout (mix of General + Modern + visuals).
    """
    layout_name = layout_name.lower().strip()
    if layout_name in ["modern", "general", "hybrid", "presenton"]:
        return HybridPresentonLayout()
    return HybridPresentonLayout()
