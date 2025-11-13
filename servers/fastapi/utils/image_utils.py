# servers/fastapi/utils/image_utils.py
"""
Pure utility image helpers used by PPTX creator.
This module purposely does NOT import any project models to avoid circular imports.
Functions accept simple values (strings, lists, dicts, or simple objects) for flexibility.
"""

from typing import Any, List, Optional, Tuple, Union
from PIL import Image, ImageOps, ImageDraw


def _resolve_object_fit(object_fit: Any) -> Tuple[Optional[str], Optional[List[float]]]:
    """
    Normalize different possible object_fit representations into (fit_mode, focus_list).
    object_fit may be:
      - None
      - a string: "contain" | "cover" | "fill"
      - a dict: {"fit": "contain", "focus": [x, y]}
      - an object with .fit and .focus attributes (enum or string)
    Returns (fit_mode, focus) where fit_mode is one of the strings or None.
    """
    if object_fit is None:
        return None, None

    # if it's a plain string
    if isinstance(object_fit, str):
        return object_fit.lower(), None

    # if it's a dict
    if isinstance(object_fit, dict):
        fit = object_fit.get("fit") or object_fit.get("mode") or object_fit.get("type")
        focus = object_fit.get("focus") or object_fit.get("focal") or None
        if isinstance(focus, (list, tuple)) and len(focus) >= 2:
            try:
                f0 = float(focus[0])
                f1 = float(focus[1])
                return (str(fit).lower() if fit else None, [f0, f1])
            except Exception:
                return (str(fit).lower() if fit else None, None)
        return (str(fit).lower() if fit else None, None)

    # if it's an object with attributes (.fit, .focus)
    fit_attr = getattr(object_fit, "fit", None)
    focus_attr = getattr(object_fit, "focus", None)
    # If fit_attr is an Enum, try to use its value or name
    if fit_attr is not None:
        fit_val = getattr(fit_attr, "value", None) or getattr(fit_attr, "name", None) or str(fit_attr)
    else:
        fit_val = None

    focus_list = None
    if focus_attr:
        try:
            focus_list = [float(focus_attr[0]), float(focus_attr[1])]
        except Exception:
            focus_list = None

    return (fit_val.lower() if isinstance(fit_val, str) else None, focus_list)


def clip_image(
    image: Image.Image,
    width: int,
    height: int,
    focus_x: float = 50.0,
    focus_y: float = 50.0,
) -> Image.Image:
    """
    Resize image maintaining aspect ratio and then crop a (width x height) region.
    focus_x and focus_y are percentages (0..100) indicating the focus point in the resized image.
    """
    img_width, img_height = image.size
    if img_width == 0 or img_height == 0 or width == 0 or height == 0:
        return image

    img_aspect = img_width / img_height
    box_aspect = width / height

    # Resize so the shorter side fits the target to ensure coverage
    if img_aspect > box_aspect:
        # image is wider -> scale height to target height
        scale = height / img_height
    else:
        # image is taller -> scale width to target width
        scale = width / img_width

    new_width = max(1, int(round(img_width * scale)))
    new_height = max(1, int(round(img_height * scale)))

    resized = image.resize((new_width, new_height), Image.LANCZOS)

    # Clamp focus
    focus_x = max(0.0, min(100.0, focus_x))
    focus_y = max(0.0, min(100.0, focus_y))

    # Determine top-left coordinates to crop so that focus point is centered as much as possible
    left = int(round((new_width - width) * (focus_x / 100.0)))
    top = int(round((new_height - height) * (focus_y / 100.0)))

    # Ensure crop box within bounds
    left = max(0, min(left, new_width - width))
    top = max(0, min(top, new_height - height))

    right = left + width
    bottom = top + height

    return resized.crop((left, top, right, bottom))


def round_image_corners(image: Image.Image, radii: Union[List[int], int]) -> Image.Image:
    """
    Apply independent corner radii.
    - radii: either a single int (applied to all corners) or a list of 4 ints [tl, tr, br, bl].
    Returns a new RGBA image with rounded corners (alpha channel applied).
    """
    if isinstance(radii, int):
        radii = [radii] * 4
    if not isinstance(radii, (list, tuple)) or len(radii) != 4:
        raise ValueError("radii must be an int or list/tuple of four ints")

    w, h = image.size
    max_radius = min(w // 2, h // 2)
    radii = [max(0, min(int(r), max_radius)) for r in radii]

    # Convert image to RGBA
    img = image.convert("RGBA")

    # Create mask
    mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(mask)

    # Draw rectangles to remove corner areas, then add quarter-circles
    # start by making full-opaque rectangle then subtract corners
    draw.rectangle((0, 0, w, h), fill=255)

    # helper to draw a corner cutout
    if radii[0] > 0:
        # top-left
        draw.pieslice((0, 0, radii[0] * 2, radii[0] * 2), 180, 270, fill=0)
        draw.rectangle((0, 0, radii[0], radii[0]), fill=0)
    if radii[1] > 0:
        # top-right
        draw.pieslice((w - radii[1] * 2, 0, w, radii[1] * 2), 270, 360, fill=0)
        draw.rectangle((w - radii[1], 0, w, radii[1]), fill=0)
    if radii[2] > 0:
        # bottom-right
        draw.pieslice((w - radii[2] * 2, h - radii[2] * 2, w, h), 0, 90, fill=0)
        draw.rectangle((w - radii[2], h - radii[2], w, h), fill=0)
    if radii[3] > 0:
        # bottom-left
        draw.pieslice((0, h - radii[3] * 2, radii[3] * 2, h), 90, 180, fill=0)
        draw.rectangle((0, h - radii[3], radii[3], h), fill=0)

    # Create result and apply mask as alpha
    result = Image.new("RGBA", (w, h))
    result.paste(img, (0, 0))
    result.putalpha(mask)
    return result


def invert_image(img: Image.Image) -> Image.Image:
    """
    Invert RGB channels, preserve alpha channel.
    """
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb_image = Image.merge("RGB", (r, g, b))
    inverted = ImageOps.invert(rgb_image)
    r2, g2, b2 = inverted.split()
    return Image.merge("RGBA", (r2, g2, b2, a))


def create_circle_image(image: Image.Image) -> Image.Image:
    """
    Make the image circular by applying a centered circular alpha mask.
    Returns an RGBA image the same size as input with transparent corners.
    """
    img = image.convert("RGBA")
    w, h = img.size
    diameter = min(w, h)

    # Create a square crop centered
    left = (w - diameter) // 2
    top = (h - diameter) // 2
    square = img.crop((left, top, left + diameter, top + diameter))

    # Build circular mask
    mask = Image.new("L", (diameter, diameter), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, diameter, diameter), fill=255)

    # Apply mask and paste into RGBA canvas of original size
    out = Image.new("RGBA", (diameter, diameter))
    out.paste(square, (0, 0))
    out.putalpha(mask)

    # If caller expects same original size, center the circular image on transparent background
    if (diameter, diameter) != (w, h):
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(out, (left, top), out)
        return canvas

    return out


def set_image_opacity(image: Image.Image, opacity: float) -> Image.Image:
    """
    Apply overall opacity multiplier (0.0 to 1.0) to image alpha channel.
    """
    opacity = max(0.0, min(1.0, float(opacity if opacity is not None else 1.0)))
    if opacity >= 1.0:
        return image

    img = image.convert("RGBA")
    r, g, b, a = img.split()
    new_alpha = a.point(lambda p: int(p * opacity))
    img.putalpha(new_alpha)
    return img


def fit_image(
    image: Image.Image,
    width: int,
    height: int,
    object_fit: Optional[Any] = None,
) -> Image.Image:
    """
    Object-fit behavior:
      - "contain": scale to fit inside the box, preserve aspect ratio, center based on focus (default focus center)
      - "cover": scale to cover the box, then crop using focus center
      - "fill": stretch to exactly width x height (no aspect preservation)

    object_fit may be string, dict, or object; this helper resolves to (fit_mode, focus).
    focus values are treated as percentages [0..100].
    """
    fit_mode, focus = _resolve_object_fit(object_fit)
    if fit_mode is None:
        return image

    # Basic guards
    if width <= 0 or height <= 0:
        return image

    img_w, img_h = image.size
    if img_w == 0 or img_h == 0:
        return image

    img_aspect = img_w / img_h
    box_aspect = width / height

    focus_x, focus_y = (50.0, 50.0)
    if isinstance(focus, (list, tuple)) and len(focus) >= 2:
        try:
            focus_x = float(focus[0])
            focus_y = float(focus[1])
        except Exception:
            pass

    # CONTAIN: scale to fit inside box, then paste on transparent canvas aligned by focus
    if fit_mode in ("contain", "scale", "fit"):
        if img_aspect > box_aspect:
            new_w = width
            new_h = int(round(width / img_aspect))
        else:
            new_h = height
            new_w = int(round(height * img_aspect))

        resized = image.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # compute paste pos so focus point controls alignment
        paste_x = int(round((width - new_w) * (focus_x / 100.0)))
        paste_y = int(round((height - new_h) * (focus_y / 100.0)))

        # clamp
        paste_x = max(0, min(paste_x, width - new_w))
        paste_y = max(0, min(paste_y, height - new_h))

        canvas.paste(resized, (paste_x, paste_y), resized.convert("RGBA"))
        return canvas

    # COVER: scale to cover and then crop around focus
    if fit_mode in ("cover", "crop"):
        if img_aspect > box_aspect:
            # image wider than box -> scale height to box height, then crop width
            scale = height / img_h
        else:
            # image taller than box -> scale width to box width, then crop height
            scale = width / img_w

        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))
        resized = image.resize((new_w, new_h), Image.LANCZOS)

        # center coords for crop using focus
        left = int(round((new_w - width) * (focus_x / 100.0)))
        top = int(round((new_h - height) * (focus_y / 100.0)))

        left = max(0, min(left, new_w - width))
        top = max(0, min(top, new_h - height))

        return resized.crop((left, top, left + width, top + height))

    # FILL: stretch
    if fit_mode in ("fill", "stretch"):
        return image.resize((width, height), Image.LANCZOS)

    # default fallback
    return image

