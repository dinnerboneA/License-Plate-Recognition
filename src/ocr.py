"""
ocr.py
OCR Text Extraction — Shared Module
CT036-3-IPPR | APU Level 3

Uses EasyOCR (PyTorch-based) to extract text from a cropped plate image.
EasyOCR is NOT TensorFlow — it satisfies the assignment restriction.
"""

import cv2
import numpy as np
import easyocr
import re

# Initialise reader once (loading model takes ~5s first time)
_reader = None

def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        # English only — Malaysian plates use Latin characters
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def _resize_for_ocr(plate: np.ndarray, target_height: int = 80) -> np.ndarray:
    """Scale plate to a standard height so filters and EasyOCR get consistent input."""
    h, w = plate.shape[:2]
    if h == 0: 
        return plate
    scale = target_height / float(h)
    return cv2.resize(plate, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)

def format_plate(text: str) -> str:
    """Formats the plate perfectly while preserving newlines for 2-row plates."""
    if not text: 
        return ""
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        # Inserts a space between letters and numbers without rearranging them
        formatted = re.sub(r'(?<=[A-Z])(?=[0-9])|(?<=[0-9])(?=[A-Z])', ' ', line)
        formatted_lines.append(formatted)
    return "\n".join(formatted_lines)

def preprocess(img: np.ndarray) -> list[np.ndarray]:
    """Generates 5 different filtered versions of the image to test."""
    # RESIZE FIRST so the kernel filters don't destroy small images!
    img = _resize_for_ocr(img)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray_clahe, -1, kernel)

    th1 = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    th2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return [gray, gray_clahe, sharp, th1, th2]

def score_plate(text: str) -> float:
    """Rates how much the text actually looks like a real license plate."""
    score = 0
    flat_text = text.replace('\n', '').replace(' ', '')
    if 4 <= len(flat_text) <= 9:
        score += 2
    if flat_text and flat_text[0].isalpha():
        score += 1
    if any(c.isdigit() for c in flat_text):
        score += 2
    if len(set(flat_text)) < len(flat_text) * 0.5:
        score -= 2
    return score

def extract_text(plate_image: np.ndarray, confidence_threshold: float = 0.4) -> tuple[str, float]:
    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    reader = _get_reader()
    processed_images = preprocess(plate_image)

    best_text = ""
    best_conf = 0.0
    best_score = -999.0

    for p in processed_images:
        for test_img in [p, cv2.bitwise_not(p)]:
            results = reader.readtext(
                test_img, 
                detail=1, 
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            if not results:
                continue

            # We REMOVED the forced left-to-right sorting here!
            # EasyOCR natively reads top-to-bottom, saving our two-row plates!
            
            accepted_parts = []
            conf_list = []
            
            for (_, txt, conf) in results:
                clean_part = re.sub(r'[^A-Z0-9]', '', txt.upper())
                if clean_part and conf >= confidence_threshold:
                    accepted_parts.append(clean_part)
                    conf_list.append(conf)

            if not accepted_parts:
                continue

            # Join parts with a newline so the State Classifier knows it's multi-line
            text = "\n".join(accepted_parts)
            avg_conf = float(np.mean(conf_list))

            # Skip junk reads (ignoring newlines in the length check)
            if len(text.replace('\n', '')) < 4:
                continue

            score = score_plate(text)
            final_score = avg_conf + (score * 0.1)

            if final_score > best_score:
                best_score = final_score
                best_conf = avg_conf
                best_text = text

    if best_text == "" or best_conf < confidence_threshold:
        return "", 0.0

    formatted_text = format_plate(best_text)
    return formatted_text, best_conf