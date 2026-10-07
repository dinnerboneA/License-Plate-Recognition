"""

Person 1 - Plate Detection / Localization Module
IPPR License Plate Recognition (LPR) and State Identification System (SIS)

What this version improves:
    - Better rejection of logos, grilles, road tiles, building grids, and bumper crops.
    - Better support for Malaysian black plates with white text.
    - Better support for compact/two-row motorcycle plates.
    - Returns top candidates so OCR can later validate candidate 1, 2, 3, etc.
    - Saves debug images for report/presentation.

This file uses classical OpenCV image processing only:
    - grayscale / HSV conversion
    - CLAHE contrast enhancement
    - thresholding
    - edge detection
    - morphology
    - connected components
    - contour analysis
    - candidate scoring

It does NOT use YOLO, TensorFlow, Haar Cascade, or template matching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

ImageInput = Union[str, np.ndarray]


@dataclass
class PlateCandidate:
    """One possible license plate region."""
    bbox: Tuple[int, int, int, int]
    box: np.ndarray
    score: float
    method: str
    text_score: float
    aspect_ratio: float
    dark_score: float
    neutral_score: float
    details: Dict[str, float]


# ============================================================
# Basic helpers
# ============================================================

def load_image(image_input: ImageInput) -> np.ndarray:
    """Loads an image path or copies an already-loaded OpenCV image."""
    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    if not os.path.exists(str(image_input)):
        raise FileNotFoundError(f"Image not found: {image_input}")

    image = cv2.imread(str(image_input))

    if image is None:
        raise FileNotFoundError(f"OpenCV could not read: {image_input}")

    return image


def resize_keep_aspect(image: np.ndarray, max_width: int = 900) -> Tuple[np.ndarray, float]:
    """Resizes large images to speed up processing and keep thresholds stable."""
    h, w = image.shape[:2]

    if w <= max_width:
        return image.copy(), 1.0

    scale = max_width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def clip_bbox(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Keeps a bounding box inside the image."""
    x, y, w, h = bbox
    H, W = image_shape[:2]

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))

    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, int],
    px: float = 0.15,
    py: float = 0.40,
    min_padding: int = 4,
) -> Tuple[int, int, int, int]:
    """Expands a text row into a fuller plate crop."""
    x, y, w, h = bbox
    pad_x = max(min_padding, int(w * px))
    pad_y = max(min_padding, int(h * py))
    return clip_bbox((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), image_shape)


def bbox_to_box(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Converts x,y,w,h to 4 corner points for drawing."""
    x, y, w, h = bbox
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.intp)


# ============================================================
# Preprocessing
# ============================================================

def preprocess(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Creates processed image versions used by several detection routes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Preserves edges better than normal blur.
    filtered = cv2.bilateralFilter(enhanced, 7, 65, 65)

    return {
        "gray": gray,
        "hsv": hsv,
        "enhanced": enhanced,
        "filtered": filtered,
    }


# ============================================================
# Text-likeness analysis
# ============================================================

def text_component_score(gray_roi: np.ndarray) -> float:
    """
    Scores whether a candidate region looks like it contains plate characters.

    This helps reject car logos, grilles, road tiles, and building grids because
    those objects may have edges but usually do not form plate-like character rows.
    """
    if gray_roi is None or gray_roi.size == 0:
        return 0.0

    h, w = gray_roi.shape[:2]

    if w < 20 or h < 8:
        return 0.0

    # Smaller normalized size keeps GUI speed acceptable.
    target_w = 220
    scale = target_w / float(max(w, 1))
    target_h = max(28, int(h * scale))

    roi = cv2.resize(gray_roi, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    binary_versions = [
        cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7),
        cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7),
    ]

    best_score = 0.0

    for binary in binary_versions:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        H, W = binary.shape[:2]
        total_area = H * W
        boxes = []

        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]

            area_ratio = area / float(total_area)
            height_ratio = ch / float(H)
            width_ratio = cw / float(W)
            comp_aspect = cw / float(max(ch, 1))

            # Remove tiny noise and huge blobs.
            if area_ratio < 0.00045 or area_ratio > 0.20:
                continue

            # Character-like size range.
            if height_ratio < 0.12 or height_ratio > 0.96:
                continue

            if width_ratio < 0.005 or width_ratio > 0.40:
                continue

            # Avoid long horizontal lines.
            if comp_aspect > 3.8:
                continue

            boxes.append((x, y, cw, ch))

        count = len(boxes)

        if count == 0:
            continue

        xs = np.array([b[0] for b in boxes])
        x2s = np.array([b[0] + b[2] for b in boxes])
        ys = np.array([b[1] for b in boxes])
        y2s = np.array([b[1] + b[3] for b in boxes])
        heights = np.array([b[3] for b in boxes], dtype=float)
        centers_y = np.array([b[1] + b[3] / 2.0 for b in boxes], dtype=float)

        coverage_x = (x2s.max() - xs.min()) / float(W)
        coverage_y = (y2s.max() - ys.min()) / float(H)

        # Normal plates usually have around 5-8 components.
        if count <= 16:
            count_score = min(count / 7.0, 1.0)
        else:
            # Too many components usually means background grid/noise.
            count_score = max(0.28, 1.0 - (count - 16) * 0.055)

        spread_score = max(min(coverage_x / 0.70, 1.0), 0.80 * min(coverage_y / 0.72, 1.0))
        height_consistency = 1.0 - min(float(np.std(heights)) / max(1.0, float(np.mean(heights))), 1.0)

        # One-row plate alignment.
        one_row_score = 1.0 - min(float(np.std(centers_y)) / max(1.0, H) / 0.22, 1.0)

        # Two-row/motorcycle plate alignment.
        two_row_score = 0.0
        if count >= 5:
            median_y = np.median(centers_y)
            upper = centers_y[centers_y <= median_y]
            lower = centers_y[centers_y > median_y]

            if len(upper) >= 2 and len(lower) >= 2:
                within_std = (np.std(upper) + np.std(lower)) / 2.0
                row_gap = abs(np.mean(lower) - np.mean(upper)) / float(H)
                two_row_score = (1.0 - min(within_std / max(1.0, H) / 0.16, 1.0)) * min(row_gap / 0.28, 1.0)

        row_score = max(one_row_score, two_row_score)

        score = (
            0.36 * count_score +
            0.28 * spread_score +
            0.18 * height_consistency +
            0.18 * row_score
        )

        best_score = max(best_score, score)

    return float(np.clip(best_score, 0.0, 1.0))


# ============================================================
# Candidate scoring
# ============================================================

def evaluate_bbox(
    image: np.ndarray,
    processed: Dict[str, np.ndarray],
    bbox: Tuple[int, int, int, int],
    method: str,
) -> Optional[PlateCandidate]:
    """Scores one possible plate bounding box. Cheap checks run before text analysis for speed."""
    x, y, w, h = clip_bbox(bbox, image.shape)

    if w <= 0 or h <= 0:
        return None

    H, W = image.shape[:2]
    area_ratio = (w * h) / float(W * H)

    if area_ratio < 0.00045 or area_ratio > 0.85:
        return None

    aspect_ratio = max(w, h) / float(max(1, min(w, h)))

    if aspect_ratio < 1.05 or aspect_ratio > 10.0:
        return None

    center_x = (x + w / 2.0) / float(W)
    center_y = (y + h / 2.0) / float(H)

    # Hard skip extreme top. Malaysian plates in these test cases are not in the top strip.
    if center_y < 0.10:
        return None

    gray_roi = processed["enhanced"][y:y + h, x:x + w]
    raw_roi = processed["gray"][y:y + h, x:x + w]
    hsv_roi = processed["hsv"][y:y + h, x:x + w]

    if gray_roi.size == 0:
        return None

    # Cheap colour/intensity checks before connected-component text analysis.
    dark_ratio = float(np.mean(raw_roi < 90))
    very_dark_ratio = float(np.mean(raw_roi < 55))
    dark_score = min(dark_ratio / 0.38, 1.0) * 0.75 + min(very_dark_ratio / 0.16, 1.0) * 0.25

    saturation = hsv_roi[:, :, 1]
    value = hsv_roi[:, :, 2]
    neutral_ratio = float(np.mean((saturation < 85) | (value < 80)))
    neutral_score = min(neutral_ratio / 0.78, 1.0)

    # Skip colourful non-plate regions early unless they are strongly dark.
    if neutral_score < 0.30 and dark_score < 0.80:
        return None

    # Skip very bottom tiny regions early: usually pavement/road tile cracks.
    if center_y > 0.88 and area_ratio < 0.018:
        return None

    p5, p95 = np.percentile(gray_roi, [5, 95])
    contrast_score = min((p95 - p5) / 115.0, 1.0)

    if contrast_score < 0.18:
        return None

    median = np.median(gray_roi)
    edges = cv2.Canny(gray_roi, int(max(0, 0.65 * median)), int(min(255, 1.35 * median)))
    edge_density = float(np.count_nonzero(edges) / edges.size)

    if edge_density < 0.0055:
        return None

    edge_score = min(edge_density / 0.12, 1.0)

    # Expensive but important text check.
    text_score = text_component_score(gray_roi)

    if text_score < 0.20:
        return None

    if 2.2 <= aspect_ratio <= 6.4:
        aspect_score = 1.0
    elif 1.65 <= aspect_ratio < 2.2:
        aspect_score = 0.82
    elif 1.05 <= aspect_ratio < 1.65:
        aspect_score = 0.66
    elif 6.4 < aspect_ratio <= 10.0:
        aspect_score = 0.58
    else:
        aspect_score = 0.30

    if center_y < 0.12:
        position_score = 0.25
    elif center_y < 0.22:
        position_score = 0.60
    elif center_y > 0.92:
        position_score = 0.35
    elif center_y > 0.84:
        position_score = 0.58
    else:
        position_score = 1.0

    width_ratio = w / float(W)
    height_ratio = h / float(H)
    size_score = min(width_ratio / 0.23, 1.0) * 0.70 + min(height_ratio / 0.08, 1.0) * 0.30

    method_bonus = {
        "white_cluster": 1.04,
        "bright_text": 1.02,
        "dark_rect": 1.00,
        "blackhat": 0.96,
        "edge": 0.86,
        "refined": 1.06,
    }.get(method, 0.88)

    score = (
        0.34 * text_score +
        0.19 * dark_score +
        0.12 * neutral_score +
        0.13 * aspect_score +
        0.09 * contrast_score +
        0.05 * edge_score +
        0.04 * position_score +
        0.02 * size_score +
        0.02 * method_bonus
    )

    if center_y > 0.86 and area_ratio < 0.015:
        score *= 0.72
    if neutral_score < 0.38 and dark_score < 0.86:
        score *= 0.68
    if area_ratio > 0.08 and text_score < 0.72:
        score *= 0.70
    if aspect_ratio < 1.45 and text_score < 0.70:
        score *= 0.76
    if method in ("bright_text", "white_cluster", "dark_rect", "blackhat", "refined") and dark_score < 0.20 and text_score < 0.78:
        score *= 0.80

    # Edge-touching crops are often partial motorcycles, road, or nearby vehicles, not the plate.
    touches_edge = x <= 3 or y <= 3 or (x + w) >= W - 3 or (y + h) >= H - 3
    if touches_edge:
        score *= 0.78

    # Motorcycle/two-row plates can be compact and higher in the image.
    if (
        method in ("blackhat", "white_cluster", "bright_text")
        and 1.30 <= aspect_ratio <= 3.50
        and h > w * 1.15
        and 0.12 <= center_y <= 0.36
        and 0.20 <= center_x <= 0.82
        and dark_score >= 0.52
        and text_score >= 0.55
    ):
        score += 0.16

    details = {
        "area_ratio": float(area_ratio),
        "edge_density": float(edge_density),
        "contrast_score": float(contrast_score),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
    }

    return PlateCandidate(
        bbox=(x, y, w, h),
        box=bbox_to_box((x, y, w, h)),
        score=float(score),
        method=method,
        text_score=float(text_score),
        aspect_ratio=float(aspect_ratio),
        dark_score=float(dark_score),
        neutral_score=float(neutral_score),
        details=details,
    )


# ============================================================
# Candidate generation routes
# ============================================================

def detect_white_clusters(image: np.ndarray, processed: Dict[str, np.ndarray]) -> List[PlateCandidate]:
    """
    Route 1: Detects bright/white character clusters.

    Good for:
        - standard Malaysian black plates with white text
        - motorcycle/two-row plates
        - small bus/van plates
    """
    hsv = processed["hsv"]
    enhanced = processed["enhanced"]
    H, W = enhanced.shape[:2]

    # White/bright neutral text.
    mask_white = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([179, 115, 255]))

    # Extra bright pixels after CLAHE.
    cut = max(145, int(np.percentile(enhanced, 80)))
    mask_bright = cv2.inRange(enhanced, cut, 255)

    mask = cv2.bitwise_or(mask_white, mask_bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    candidates: List[PlateCandidate] = []

    # Kernels for normal row, compact/two-row, and general text clusters.
    configs = [
        ((19, 5), 1, 0.35, 0.85),
        ((21, 21), 1, 0.30, 0.45),
        ((31, 15), 1, 0.20, 0.35),
    ]

    for kernel_size, iterations, px, py in configs:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 18 or h < 8:
                continue

            raw_center_y = (y + h / 2.0) / float(H)
            if raw_center_y < 0.10 or raw_center_y > 0.90:
                continue

            if w / float(W) < 0.018 or h / float(H) < 0.004:
                continue

            rough_aspect = max(w, h) / float(max(1, min(w, h)))
            if rough_aspect > 12:
                continue

            bbox = expand_bbox((x, y, w, h), image.shape, px=px, py=py, min_padding=4)
            candidate = evaluate_bbox(image, processed, bbox, "white_cluster")

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def detect_bright_text_rows(image: np.ndarray, processed: Dict[str, np.ndarray]) -> List[PlateCandidate]:
    """Route 2: Groups bright text rows using morphology."""
    enhanced = processed["enhanced"]
    H, W = enhanced.shape[:2]

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    cut = max(135, int(np.percentile(blur, 78)))
    mask = cv2.inRange(blur, cut, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    candidates: List[PlateCandidate] = []

    for kernel_size in [(15, 3), (25, 5), (18, 18)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 20 or h < 8:
                continue

            raw_center_y = (y + h / 2.0) / float(H)
            if raw_center_y < 0.10 or raw_center_y > 0.90:
                continue

            rough_aspect = max(w, h) / float(max(1, min(w, h)))
            if rough_aspect > 12:
                continue

            bbox = expand_bbox((x, y, w, h), image.shape, px=0.22, py=0.75, min_padding=4)
            candidate = evaluate_bbox(image, processed, bbox, "bright_text")

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def detect_dark_rectangles(image: np.ndarray, processed: Dict[str, np.ndarray]) -> List[PlateCandidate]:
    """Route 3: Finds dark rectangular regions that may contain plate text."""
    gray = processed["gray"]
    H, W = gray.shape[:2]

    limit = max(55, min(112, int(np.percentile(gray, 30))))
    _, mask = cv2.threshold(gray, limit, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    candidates: List[PlateCandidate] = []

    for kernel_size in [(25, 7), (17, 13), (21, 21)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 25 or h < 10:
                continue

            raw_center_y = (y + h / 2.0) / float(H)
            if raw_center_y < 0.10 or raw_center_y > 0.90:
                continue

            bbox = expand_bbox((x, y, w, h), image.shape, px=0.05, py=0.10, min_padding=3)
            candidate = evaluate_bbox(image, processed, bbox, "dark_rect")

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def detect_blackhat_regions(image: np.ndarray, processed: Dict[str, np.ndarray]) -> List[PlateCandidate]:
    """Route 4: Blackhat morphology for bright text on dark backgrounds."""
    gray = processed["filtered"]
    H, W = gray.shape[:2]
    candidates: List[PlateCandidate] = []

    for kernel_size in [(25, 7), (17, 13)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        grad = np.absolute(grad)

        min_val, max_val = np.min(grad), np.max(grad)

        if max_val - min_val < 1e-6:
            continue

        grad = (255 * ((grad - min_val) / (max_val - min_val))).astype("uint8")
        grad = cv2.GaussianBlur(grad, (5, 5), 0)

        connected = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, thresh = cv2.threshold(connected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 25 or h < 10:
                continue

            raw_center_y = (y + h / 2.0) / float(H)
            if raw_center_y < 0.10 or raw_center_y > 0.90:
                continue

            bbox = expand_bbox((x, y, w, h), image.shape, px=0.08, py=0.18, min_padding=4)
            candidate = evaluate_bbox(image, processed, bbox, "blackhat")

            if candidate is not None:
                candidates.append(candidate)

    return candidates


def detect_edge_regions(image: np.ndarray, processed: Dict[str, np.ndarray]) -> List[PlateCandidate]:
    """Route 5: Edge and contour fallback."""
    gray = processed["filtered"]
    H, W = gray.shape[:2]
    median = np.median(gray)
    edges = cv2.Canny(gray, int(max(0, 0.67 * median)), int(min(255, 1.33 * median)))

    candidates: List[PlateCandidate] = []

    for kernel_size in [(25, 5), (15, 9)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 30 or h < 12:
                continue

            raw_center_y = (y + h / 2.0) / float(H)
            if raw_center_y < 0.10 or raw_center_y > 0.90:
                continue

            bbox = expand_bbox((x, y, w, h), image.shape, px=0.07, py=0.12, min_padding=4)
            candidate = evaluate_bbox(image, processed, bbox, "edge")

            if candidate is not None:
                candidates.append(candidate)

    return candidates


# ============================================================
# Candidate refinement and selection
# ============================================================

def non_max_suppression(candidates: List[PlateCandidate], iou_threshold: float = 0.34) -> List[PlateCandidate]:
    """Removes heavily overlapping duplicate candidates."""
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: List[PlateCandidate] = []

    def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter

        return inter / union if union > 0 else 0.0

    for candidate in candidates:
        if all(iou(candidate.bbox, old.bbox) <= iou_threshold for old in kept):
            kept.append(candidate)

    return kept


def refine_candidate(image: np.ndarray, processed: Dict[str, np.ndarray], candidate: PlateCandidate) -> List[PlateCandidate]:
    """
    Tightens loose candidates.

    Example:
        If the first crop contains bumper + plate, this searches inside the crop
        for a tighter plate-sized region.
    """
    x, y, w, h = candidate.bbox

    # Already tight and strong: no need to refine.
    if candidate.details["area_ratio"] < 0.012 and candidate.text_score > 0.72:
        return []

    roi = image[y:y + h, x:x + w]

    if roi.size == 0:
        return []

    roi_processed = preprocess(roi)
    sub_candidates: List[PlateCandidate] = []
    sub_candidates.extend(detect_white_clusters(roi, roi_processed))
    sub_candidates.extend(detect_bright_text_rows(roi, roi_processed))
    sub_candidates.extend(detect_dark_rectangles(roi, roi_processed))
    sub_candidates = non_max_suppression(sub_candidates, 0.32)

    refined: List[PlateCandidate] = []

    for sub in sub_candidates[:4]:
        sx, sy, sw, sh = sub.bbox

        # If the sub-crop is basically the whole loose candidate, it is not useful.
        if sw * sh > 0.80 * w * h and candidate.details["area_ratio"] > 0.02:
            continue

        global_bbox = clip_bbox((x + sx, y + sy, sw, sh), image.shape)
        refined_candidate = evaluate_bbox(image, processed, global_bbox, "refined")

        if refined_candidate is not None:
            # Small bonus only if it is genuinely tighter or more text-like.
            if refined_candidate.details["area_ratio"] < candidate.details["area_ratio"] * 0.85 or refined_candidate.text_score > candidate.text_score + 0.04:
                refined_candidate.score += 0.035
            refined.append(refined_candidate)

    return refined


def choose_best_candidate(candidates: List[PlateCandidate]) -> Optional[PlateCandidate]:
    """
    Chooses final best candidate after sorting.

    Keeps the top candidates too, so OCR can later test candidate 1, 2, 3.
    """
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    best = candidates[0]

    # Avoid choosing road tile/background at the very bottom when a close candidate exists.
    if best.details["center_y"] > 0.86:
        for c in candidates[1:12]:
            if c.details["center_y"] < 0.84 and c.score >= best.score - 0.18 and c.text_score >= 0.55:
                return c

    # If the top candidate is a credible vertical motorcycle/two-row plate, keep it.
    bx, by, bw, bh = best.bbox
    if (
        bh > bw * 1.15
        and best.details["center_y"] < 0.36
        and best.text_score >= 0.55
        and best.dark_score >= 0.50
    ):
        return best

    # Prefer tight plate over loose bumper/front crop if score is close.
    # Important: only do this when the current best is actually loose/large.
    if best.details["area_ratio"] > 0.025:
        for c in candidates[1:12]:
            close_score = c.score >= best.score - 0.10
            much_tighter = c.details["area_ratio"] < best.details["area_ratio"] * 0.70
            plate_like = 1.35 <= c.aspect_ratio <= 6.8 and c.text_score >= 0.56 and c.dark_score >= 0.20
            not_bottom = c.details["center_y"] < 0.86

            if close_score and much_tighter and plate_like and not_bottom:
                return c

    # Prefer dark/text plate candidates over signs/background if close.
    for c in candidates[:8]:
        if c.dark_score >= 0.45 and c.text_score >= 0.70 and 1.20 <= c.aspect_ratio <= 6.8 and c.details["center_y"] < 0.86:
            if c.score >= best.score - 0.12:
                return c

    return best


# ============================================================
# Output preparation
# ============================================================

def prepare_plate_for_ocr(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Creates a cleaned plate image for the OCR module."""
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    if w < 480:
        scale = 480 / float(max(1, w))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    enhanced = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    plate_for_ocr = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        7,
    )

    return plate_for_ocr


def candidate_to_dict(candidate: PlateCandidate) -> Dict[str, object]:
    """Converts a candidate to plain Python values for GUI/report usage."""
    return {
        "bbox": candidate.bbox,
        "score": round(candidate.score, 4),
        "method": candidate.method,
        "text_score": round(candidate.text_score, 3),
        "aspect_ratio": round(candidate.aspect_ratio, 3),
        "dark_score": round(candidate.dark_score, 3),
        "neutral_score": round(candidate.neutral_score, 3),
        "details": candidate.details,
    }


def draw_debug_candidates(image: np.ndarray, candidates: List[PlateCandidate], top_k: int = 8) -> np.ndarray:
    """Draws top candidate boxes with scores."""
    debug = image.copy()

    for i, candidate in enumerate(candidates[:top_k], start=1):
        cv2.drawContours(debug, [candidate.box], -1, (0, 255, 0), 2)
        x, y, w, h = candidate.bbox
        label = (
            f"{i}: {candidate.score:.2f} {candidate.method} "
            f"T:{candidate.text_score:.2f} D:{candidate.dark_score:.2f} AR:{candidate.aspect_ratio:.1f}"
        )
        cv2.putText(debug, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    return debug


def save_debug_outputs(result: Dict[str, object], output_dir: Optional[str]) -> None:
    """Saves output images for testing and report screenshots."""
    out = Path(output_dir or "plate_detection_outputs")
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "01_image_with_best_box.jpg"), result["image_with_box"])
    cv2.imwrite(str(out / "04_top_candidates_debug.jpg"), result["candidates_debug"])

    if result["plate_crop"] is not None:
        cv2.imwrite(str(out / "02_plate_crop.jpg"), result["plate_crop"])

    if result["plate_for_ocr"] is not None:
        cv2.imwrite(str(out / "03_plate_for_ocr.jpg"), result["plate_for_ocr"])

    crops_dir = out / "top_candidate_crops"
    crops_dir.mkdir(exist_ok=True)

    resized_image = result["resized_image"]

    for i, candidate in enumerate(result["_candidate_objects"], start=1):
        x, y, w, h = candidate.bbox
        crop = resized_image[y:y + h, x:x + w]

        if crop.size == 0:
            continue

        filename = f"candidate_{i:02d}_score_{candidate.score:.2f}_{candidate.method}.jpg"
        cv2.imwrite(str(crops_dir / filename), crop)


# ============================================================
# Main reusable function
# ============================================================

def detect_plate(
    image_input: ImageInput,
    max_width: int = 1000,
    debug: bool = False,
    output_dir: Optional[str] = None,
    top_k: int = 8,
) -> Dict[str, object]:
    """
    Main function used by the group and by the future GUI.

    Example:
        from plate_detector import detect_plate
        result = detect_plate("test_images/car_01.jpg", debug=True)

    Returns:
        result["plate_crop"]       -> cropped plate image
        result["plate_for_ocr"]    -> cleaned crop for OCR
        result["image_with_box"]   -> original image with selected box
        result["top_candidates"]   -> list of candidate metadata
    """
    original = load_image(image_input)
    image, scale = resize_keep_aspect(original, max_width=max_width)
    processed = preprocess(image)

    candidates: List[PlateCandidate] = []
    candidates.extend(detect_white_clusters(image, processed))
    candidates.extend(detect_bright_text_rows(image, processed))
    candidates.extend(detect_dark_rectangles(image, processed))
    candidates.extend(detect_blackhat_regions(image, processed))
    candidates.extend(detect_edge_regions(image, processed))

    candidates = non_max_suppression(candidates, iou_threshold=0.34)
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Refinement is kept as a function above, but disabled by default here for speed.
    # The OCR module can later validate top_candidates[0], top_candidates[1], etc.
    best = choose_best_candidate(candidates)

    image_with_box = image.copy()

    if best is not None:
        cv2.drawContours(image_with_box, [best.box], -1, (0, 255, 0), 3)
        x, y, w, h = best.bbox
        plate_crop = image[y:y + h, x:x + w].copy()
        plate_for_ocr = prepare_plate_for_ocr(plate_crop)
        bbox_original_scale = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
    else:
        plate_crop = None
        plate_for_ocr = None
        bbox_original_scale = None

    candidates_debug = draw_debug_candidates(image, candidates, top_k=top_k)

    result: Dict[str, object] = {
        "success": best is not None,
        "message": "Plate candidate detected." if best is not None else "No plate candidate found.",
        "original_image": original,
        "resized_image": image,
        "image_with_box": image_with_box,
        "candidates_debug": candidates_debug,
        "plate_crop": plate_crop,
        "plate_for_ocr": plate_for_ocr,
        "bbox": best.bbox if best is not None else None,
        "bbox_original_scale": bbox_original_scale,
        "confidence": round(best.score, 4) if best is not None else 0.0,
        "method": best.method if best is not None else None,
        "candidates_count": len(candidates),
        "top_candidates": [candidate_to_dict(c) for c in candidates[:top_k]],
        "_candidate_objects": candidates[:top_k],
    }

    if debug:
        save_debug_outputs(result, output_dir)

    return result


# ============================================================
# Batch testing helper
# ============================================================

def batch_test_folder(
    input_folder: str = "test_images",
    output_root: str = "plate_detection_outputs",
) -> None:
    """Tests every .jpg/.jpeg/.png image in a folder."""
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"Folder not found: {input_folder}")
        print("Create a folder called test_images and place your images inside it.")
        return

    image_files: List[Path] = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(input_path.glob(ext))

    image_files = sorted(image_files)

    print(f"Found {len(image_files)} image(s).")

    if not image_files:
        print("No images found. Use .jpg, .jpeg, or .png")
        return

    for image_file in image_files:
        safe_name = image_file.stem.replace(" ", "_").replace("(", "").replace(")", "")
        output_dir = Path(output_root) / safe_name

        print(f"\nTesting: {image_file.name}")

        result = detect_plate(
            image_input=str(image_file),
            max_width=600,
            debug=True,
            output_dir=str(output_dir),
            top_k=8,
        )

        print("Success:", result["success"])
        print("Method:", result["method"])
        print("Confidence:", result["confidence"])
        print("BBox:", result["bbox"])
        print("Output:", output_dir)

    print(f"\nDone. Check the {output_root} folder.")


if __name__ == "__main__":
    batch_test_folder(
        input_folder="test_images",
        output_root="plate_detection_outputs",
    )
