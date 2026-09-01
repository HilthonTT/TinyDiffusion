"""Rendering a sample grid as terminal cells.

The most interesting thing a training run produces is the pictures, and a
display that could show every number about them except the images themselves
would be missing the point. A terminal cell is about twice as tall as it is
wide, so one character carries two pixels: the upper half block ``▀`` painted
with the top pixel as its foreground and the bottom as its background. The
result is square pixels at the resolution the terminal can actually manage,
which for a 32px MNIST grid is more than enough to see a 7 become a 7.

Deliberately free of both Textual and Rich. It returns colours, not markup, so
it can be tested on an install that has neither — and so the one piece here
with any arithmetic in it is not wedged behind an optional import.
"""

from pathlib import Path

__all__ = ["HALF_BLOCK", "Rgb", "Row", "half_block_rows"]

HALF_BLOCK = "▀"
"""The glyph the rows are meant to be drawn with: top pixel over bottom pixel."""

Rgb = tuple[int, int, int]
"""A colour, as the 0-255 triple every terminal palette wants."""

Row = list[tuple[Rgb, Rgb]]
"""One line of cells, each the ``(top, bottom)`` colours of a :data:`HALF_BLOCK`."""


def half_block_rows(path: Path, *, max_width: int = 64, max_height: int = 32) -> list[Row]:
    """Read an image and reduce it to half-block cells that fit the given box.

    The image is scaled to fit inside ``max_width`` columns and ``max_height``
    rows while keeping its aspect ratio, counting two image pixels per row —
    so a square image comes back in half as many rows as columns, and looks
    square on screen rather than squashed.

    Args:
        path: the PNG to read, typically a grid written by
            :func:`~tinydiffusion.training.artifacts.save_samples`.
        max_width: columns available. The result is never wider.
        max_height: rows available. The result is never taller.

    Returns:
        One list per terminal row, each holding the ``(top, bottom)`` colour of
        every cell in it. Empty if the box has no room, or the image no pixels.

    Raises:
        OSError: if the file cannot be read as an image.
    """
    if max_width < 1 or max_height < 1:
        return []

    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")

        if not image.width or not image.height:
            return []

        scale = min(max_width / image.width, (max_height * 2) / image.height, 1.0)
        width = max(int(image.width * scale), 1)
        height = max(int(image.height * scale), 1)
        height += height % 2

        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        raw = resized.tobytes()

    stride = width * 3
    rows: list[Row] = []
    for y in range(0, height, 2):
        top = raw[y * stride : (y + 1) * stride]
        bottom = raw[(y + 1) * stride : (y + 2) * stride]
        rows.append(
            [
                (
                    (top[x * 3], top[x * 3 + 1], top[x * 3 + 2]),
                    (bottom[x * 3], bottom[x * 3 + 1], bottom[x * 3 + 2]),
                )
                for x in range(width)
            ]
        )
    return rows
