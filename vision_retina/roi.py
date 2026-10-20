"""Semantic-safe crop helpers.

The incremental bus hands back dirty rectangles that come straight from the
capture backend.  Those rectangles are pixel accurate but have no idea what a
glyph or a label is: cropping exactly on the boundary can cut a clock digit in
half ("21:3" -> "53") and make the vision model answer nonsense.

This module expands a dirty rectangle until it stops cutting content:

* build a foreground mask from local contrast (works for light-on-dark and
  dark-on-light, and for coloured UI);
* group connected components into text-like lines (vertical overlap + small
  horizontal gap);
* grow the rectangle to fully contain every line/component it touches.

The result is either a safe rectangle, or (when the expansion would swallow
most of the screen) ``None`` so the caller can fall back to the full frame.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .frames import Rect


def foreground_mask(pixels: np.ndarray, blur: int = 31, threshold: int = 16) -> np.ndarray:
    """Return a uint8 mask of locally-contrasting pixels (text, icons, edges)."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    k = blur if blur % 2 == 1 else blur + 1
    background = cv2.medianBlur(gray, k)
    diff = cv2.absdiff(gray, background)
    _, mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask.astype(np.uint8)


def _component_boxes(mask: np.ndarray, min_area: int = 4) -> list[tuple[int, int, int, int]]:
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area and w >= 2 and h >= 2:
            boxes.append((x, y, w, h))
    return boxes


def group_lines(
    boxes: Iterable[tuple[int, int, int, int]],
    max_gap_x: float = 10.0,
    min_overlap: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    """Merge component boxes into lines: same row, small horizontal gap."""
    lines: list[tuple[int, int, int, int]] = []
    for x, y, w, h in sorted(boxes, key=lambda b: (b[1], b[0])):
        merged = False
        for i, (lx, ly, lw, lh) in enumerate(lines):
            overlap = min(y + h, ly + lh) - max(y, ly)
            if overlap <= 0 or overlap < min_overlap * min(h, lh):
                continue
            gap = max(x - (lx + lw), lx - (x + w))
            # gap between characters of one label is small; a new word/far
            # control should stay a separate line.
            if gap <= max(max_gap_x, 0.5 * max(h, lh)):
                nx = min(lx, x)
                ny = min(ly, y)
                nx2 = max(lx + lw, x + w)
                ny2 = max(ly + lh, y + h)
                lines[i] = (nx, ny, nx2 - nx, ny2 - ny)
                merged = True
                break
        if not merged:
            lines.append((x, y, w, h))
    return lines


def _intersects(a: Rect, b: Rect) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)


def _contains(outer: Rect, inner: Rect) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.x + outer.w >= inner.x + inner.w
        and outer.y + outer.h >= inner.y + inner.h
    )


def expand_rect_to_content(
    pixels: np.ndarray,
    rect: Rect,
    *,
    margin: int = 6,
    max_area_ratio: float = 0.62,
    max_passes: int = 6,
) -> tuple[Rect, bool]:
    """Grow ``rect`` until no glyph/line is cut by its boundary.

    Returns ``(rect, safe)``.  ``safe`` is False when the requested rectangle
    could not be made safe without exceeding ``max_area_ratio`` — the caller
    should use the full frame in that case.
    """
    h, w = pixels.shape[:2]
    r = rect.clip(w, h)
    if r.area <= 0:
        return Rect(0, 0, w, h), False

    mask = foreground_mask(pixels)
    lines = group_lines(_component_boxes(mask))
    for _ in range(max_passes):
        grown = r
        for lx, ly, lw, lh in lines:
            line = Rect(lx, ly, lw, lh)
            if _intersects(grown, line) and not _contains(grown, line):
                x0 = min(grown.x, line.x)
                y0 = min(grown.y, line.y)
                x1 = max(grown.x + grown.w, line.x + line.w)
                y1 = max(grown.y + grown.h, line.y + line.h)
                grown = Rect(x0, y0, x1 - x0, y1 - y0)
        if grown == r:
            break
        r = grown
        if r.area > max_area_ratio * w * h:
            return Rect(0, 0, w, h), False

    if margin:
        r = Rect(r.x - margin, r.y - margin, r.w + 2 * margin, r.h + 2 * margin).clip(w, h)
        if r.area > max_area_ratio * w * h:
            return Rect(0, 0, w, h), False

    # final audit: any line still partially inside?
    for lx, ly, lw, lh in lines:
        line = Rect(lx, ly, lw, lh)
        if _intersects(r, line) and not _contains(r, line):
            if r.area * 1.4 > max_area_ratio * w * h:
                return Rect(0, 0, w, h), False

    return r, True


def safe_crop_bounds(
    pixels: np.ndarray,
    rect: Rect | None,
    *,
    max_area_ratio: float = 0.62,
) -> tuple[Rect | None, bool]:
    """Convenience wrapper: ``None`` rect means "nothing changed" -> full frame."""
    if rect is None:
        return None, True
    return expand_rect_to_content(pixels, rect, max_area_ratio=max_area_ratio)
