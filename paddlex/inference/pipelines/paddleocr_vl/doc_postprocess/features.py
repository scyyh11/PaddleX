# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image measurements of OCR blocks used by the post-processing models."""

import numpy as np

FONT_LABELS = frozenset(
    {
        "text",
        "paragraph_title",
        "doc_title",
        "header",
        "figure_title",
        "aside_text",
        "vertical_text",
    }
)
LINE_LABELS = frozenset(
    {
        "text",
        "abstract",
        "reference_content",
        "vertical_text",
        "content",
        "paragraph_title",
        "doc_title",
        "algorithm",
        "code",
        "code_txt",
    }
)
LABEL_LABELS = frozenset(
    {"figure_title", "vision_footnote", "footnote", "header", "footer"}
)
_LONG_BLOCKS_PER_PAGE = 3
_KERNEL = np.ones((3, 3), np.uint8)

# libpng's RGB -> gray conversion (used by cv2.imread(..., IMREAD_GRAYSCALE) for PNG files):
# (9797 R + 19234 G + 3737 B) >> 15.
_PNG_GRAY = np.array(
    [[3737 / 32768, 19234 / 32768, 9797 / 32768, -0.5 + 2.0**-16]], np.float32
)
_transform_is_exact = None


def _png_gray_reference(bgr):
    b, g, r = (bgr[:, :, i].astype(np.uint32) for i in range(3))
    return ((r * 9797 + g * 19234 + b * 3737) >> 15).astype(np.uint8)


def _check_transform():
    """True when cv2.transform reproduces the integer conversion (checked once)."""
    import cv2

    rng = np.random.default_rng(0)
    sample = rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)
    values = np.arange(256, dtype=np.uint8)
    sample[0, :, 0] = sample[1, :, 1] = sample[2, :, 2] = values
    sample[3] = np.stack([values] * 3, axis=1)
    out = cv2.transform(sample, _PNG_GRAY)
    out = out[:, :, 0] if out.ndim == 3 else out
    return bool(np.array_equal(out, _png_gray_reference(sample)))


def to_gray(image):
    """Convert a BGR, BGRA, or single-channel array to grayscale uint8 matching PNG IMREAD_GRAYSCALE."""
    import cv2

    global _transform_is_exact
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    bgr = np.ascontiguousarray(image[:, :, :3])
    if _transform_is_exact is None:
        _transform_is_exact = _check_transform()
    if not _transform_is_exact:
        return _png_gray_reference(bgr)
    out = cv2.transform(bgr, _PNG_GRAY)
    return out[:, :, 0] if out.ndim == 3 else out


def _bounds(shape, box, ow, oh):
    """Pixel rows and columns (y0, y1, x0, x1) of a block box given in page coordinates."""
    h, w = shape
    x, y, xx, yy = [
        round(float(v) * scale)
        for v, scale in zip(box, [w / ow, h / oh, w / ow, h / oh])
    ]
    return max(0, y), min(h, yy), max(0, x), min(w, xx)


def _crop(gray, box, ow, oh):
    y0, y1, x0, x1 = _bounds(gray.shape, box, ow, oh)
    return gray[y0:y1, x0:x1]


def _ink(crop):
    """Binary ink mask without specks (area < 3, height < 2) and rules (width >= max(30, 16 h))."""
    import cv2

    _, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    width, height, area = stats[:, 2], stats[:, 3], stats[:, 4]
    drop = (area < 3) | (height < 2) | (width >= np.maximum(30, height * 16))
    drop[0] = False
    dropped = np.flatnonzero(drop)
    if not len(dropped):
        return ink
    if len(dropped) * 4000 > ink.size:
        keep = (~drop).astype(np.uint8)
        keep[0] = 0
        return keep[labels]
    for i in dropped.tolist():
        x, y, cw, ch = stats[i, :4].tolist()
        region = ink[y : y + ch, x : x + cw]
        region[labels[y : y + ch, x : x + cw] == i] = 0
    return ink


def _line_spans(ink):
    """Row ranges [a, b) of text lines from horizontal ink projection, or None."""
    import cv2

    profile = cv2.reduce(ink, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel()
    rows = np.flatnonzero(profile >= max(2, ink.shape[1] * 0.003))
    if len(rows) < 3:
        return None
    breaks = np.flatnonzero(np.diff(rows) > 2)
    starts = np.concatenate((rows[:1], rows[breaks + 1]))
    ends = np.concatenate((rows[breaks], rows[-1:])) + 1
    tall = ends - starts >= 3
    return list(zip(starts[tall].tolist(), ends[tall].tolist()))


def _line_bounds(ink, spans):
    """[first column, last column + 1, a, b] and ink pixel count of every line."""
    import cv2

    bounds, counts = [], []
    for a, b in spans:
        columns = cv2.reduce(ink[a:b], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel()
        cols = np.flatnonzero(columns)
        if len(cols):
            bounds.append([int(cols[0]), int(cols[-1]) + 1, a, b])
            counts.append(int(columns.sum()))
    return bounds, counts


def _font_values(ink, spans, bounds, counts, page_width):
    import cv2

    runs = [b - a for a, b in spans]
    font_px = float(np.median(runs))
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    ridge = (distance >= cv2.dilate(distance, _KERNEL) - 0.001) & (distance > 0)
    stroke = float(np.mean(distance[ridge])) if ridge.any() else 0
    area = sum((b - a) * (x2 - x1) for x1, x2, a, b in bounds)
    return {
        "font": font_px / page_width * 1000,
        "stroke": stroke / font_px,
        "density": sum(counts) / max(area, 1),
        "raw_density": cv2.countNonZero(ink) / ink.size,
        "lines": len(runs),
        "font_pixels": font_px,
    }


def _line_values(ink, spans, bounds, rotated):
    if not bounds:
        return None
    font = float(np.median([b - a for a, b in spans]))
    widths = [bb[1] - bb[0] for bb in bounds]
    left = [bb[0] for bb in bounds]
    right = [bb[1] for bb in bounds]
    scale = max(font, 1)
    return {
        "rotated": rotated,
        "font": font,
        "lines": len(bounds),
        "first_left": left[0] / scale,
        "first_right": (ink.shape[1] - right[0]) / scale,
        "last_left": left[-1] / scale,
        "last_right": (ink.shape[1] - right[-1]) / scale,
        "median_left": float(np.median(left)) / scale,
        "median_right": (ink.shape[1] - float(np.median(right))) / scale,
        "extent": sum(widths) / scale,
        "bounds": bounds,
        "crop_width": ink.shape[1],
        "crop_height": ink.shape[0],
    }


def measure_font(gray, box, ow, oh):
    """Typography of one heading candidate block, or None for a crop without text lines."""
    return _measure_font(gray, box, ow, oh)[0]


def _measure_font(gray, box, ow, oh):
    """(heading features, line features) of one block; line features are False for vertical crops."""
    crop = _crop(gray, box, ow, oh)
    if min(crop.shape, default=0) < 3:
        return None, None
    ink = _ink(crop)
    spans = _line_spans(ink)
    rotated = crop.shape[0] > crop.shape[1] * 3
    if not spans:
        return None, (False if rotated else None)
    bounds, counts = _line_bounds(ink, spans)
    font = _font_values(ink, spans, bounds, counts, gray.shape[1])
    return font, (False if rotated else _line_values(ink, spans, bounds, False))


def measure_lines(gray, box, ow, oh):
    """Line layout of one block (vertical blocks are measured rotated), or None."""
    crop = _crop(gray, box, ow, oh)
    if min(crop.shape, default=0) < 3:
        return None
    rotated = crop.shape[0] > crop.shape[1] * 3
    if rotated:
        crop = np.rot90(crop, -1)
    ink = _ink(np.ascontiguousarray(crop) if rotated else crop)
    spans = _line_spans(ink)
    if not spans:
        return None
    bounds, _ = _line_bounds(ink, spans)
    return _line_values(ink, spans, bounds, rotated)


class Measurements:
    """Per-document cache of non-vertical line features keyed by (page_idx, y0, y1, x0, x1)."""

    def __init__(self):
        self.lines = {}


def _map(function, items, executor):
    """``map`` in order, optionally on ``executor`` threads (OpenCV releases the GIL)."""
    if executor is None or len(items) < 2:
        return list(map(function, items))
    return list(executor.map(function, items))


def _heading_candidates(page):
    """[(block index, block)] of the blocks measured for the heading features of one page."""
    selected = []
    long_blocks = 0
    for bi, block in enumerate(page["parsing_res_list"]):
        content = block["block_content"]
        if block["block_label"] not in FONT_LABELS or not content.strip():
            continue
        if len(content) > 350:
            if long_blocks >= _LONG_BLOCKS_PER_PAGE:
                continue
            long_blocks += 1
        selected.append((bi, block))
    return selected


def measure_page_headings(page, image, candidates=None):
    """Heading measurements of one page with its grayscale image."""
    ow, oh = page["width"], page["height"]
    if candidates is None:
        candidates = _heading_candidates(page)
    return [
        (bi, _bounds(image.shape, block["block_bbox"], ow, oh))
        + _measure_font(image, block["block_bbox"], ow, oh)
        for bi, block in candidates
    ]


def heading_features(pages, gray, measurements=None, executor=None, measured=None):
    """Heading features of the OCR pages: ``{'page:block': values}``."""

    def measure(pi):
        if measured is not None and measured[pi] is not None:
            return measured[pi]
        candidates = _heading_candidates(pages[pi])
        if not candidates:
            return []
        image = gray(pi)
        if image is None:
            return []
        return measure_page_headings(pages[pi], image, candidates)

    indices = list(range(len(pages)))
    features = {}
    for pi, items in zip(indices, _map(measure, indices, executor)):
        for bi, bounds, value, lines in items:
            if measurements is not None and lines is not False:
                measurements.lines[(pi,) + bounds] = lines
            if value:
                features[f"{pi}:{bi}"] = value
    return features


def line_features(
    prepared, gray, text_features, label_features, measurements=None, executor=None
):
    """Line features of the prepared pages added to ``text_features`` and ``label_features``."""
    work = []
    for pi, page in enumerate(prepared):
        selected = [
            (bi, block)
            for bi, block in enumerate(page["parsing_res_list"])
            if block["block_label"] in LINE_LABELS
            or block["block_label"] in LABEL_LABELS
        ]
        if selected:
            work.append((pi, page, selected))
    known = {} if measurements is None else measurements.lines

    def measure(item):
        pi, page, selected = item
        image = gray(pi)
        if image is None:
            return []
        ow, oh = page["width"], page["height"]
        values = []
        for bi, block in selected:
            box = block["block_bbox"]
            key = (pi,) + _bounds(image.shape, box, ow, oh)
            if key in known:
                value = known[key]
            else:
                value = measure_lines(image, box, ow, oh)
            values.append((bi, block["block_label"], value))
        return values

    for (pi, _, _), measured in zip(work, _map(measure, work, executor)):
        for bi, label, value in measured:
            if value:
                target = text_features if label in LINE_LABELS else label_features
                target[f"{pi}:{bi}"] = value
