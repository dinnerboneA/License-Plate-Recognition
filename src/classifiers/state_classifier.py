"""
state_classifier.py
Person 3 — State Identification Module
CT036-3-IPPR | APU Level 3

Maps Malaysian vehicle plate prefix to registered state/territory.
Handles single-row (car) and two-row (motorcycle) plate formats.

Interface contract (shared with all group classifiers):
    classify(plate_text: str) -> dict | None
    Returns None if plate text cannot be identified.
"""

import re

# ---------------------------------------------------------------------------
# State prefix lookup table
# Maps plate prefix -> (state_name, full_name)
# ---------------------------------------------------------------------------
PREFIX_MAP = {
    # Federal Territories
    "W":  ("Kuala Lumpur", "Wilayah Persekutuan Kuala Lumpur"),
    "V":  ("Kuala Lumpur", "Wilayah Persekutuan Kuala Lumpur (Overflow)"),
    "F":  ("Putrajaya",    "Wilayah Persekutuan Putrajaya"),
    "H":  ("Labuan",       "Wilayah Persekutuan Labuan"),

    # Peninsular Malaysia
    "A":  ("Perak",            "Negeri Perak Darul Ridzuan"),
    "B":  ("Selangor",         "Negeri Selangor Darul Ehsan"),
    "C":  ("Pahang",           "Negeri Pahang Darul Makmur"),
    "D":  ("Kelantan",         "Negeri Kelantan Darul Naim"),
    "J":  ("Johor",            "Negeri Johor Darul Takzim"),
    "K":  ("Kedah",            "Negeri Kedah Darul Aman"),
    "M":  ("Melaka",           "Negeri Melaka Bandaraya Bersejarah"),
    "N":  ("Negeri Sembilan",  "Negeri Sembilan Darul Khusus"),
    "P":  ("Pulau Pinang",     "Negeri Pulau Pinang"),
    "R":  ("Perlis",           "Negeri Perlis Indera Kayangan"),
    "T":  ("Terengganu",       "Negeri Terengganu Darul Iman"),

    # East Malaysia
    "Q":  ("Sabah",   "Negeri Sabah"),
    "S":  ("Sarawak", "Negeri Sarawak"),

    # Multi-letter overflow series (states that exhausted single letters)
    "AB": ("Johor",          "Negeri Johor Darul Takzim (Series 2)"),
    "BA": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 2)"),
    "BB": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 3)"),
    "BC": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 4)"),
    "BD": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 5)"),
    "BE": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 6)"),
    "BF": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 7)"),
    "BG": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 8)"),
    "BH": ("Selangor",       "Negeri Selangor Darul Ehsan (Series 9)"),
    "CA": ("Pahang",         "Negeri Pahang Darul Makmur (Series 2)"),
    "DA": ("Kelantan",       "Negeri Kelantan Darul Naim (Series 2)"),
    "JA": ("Johor",          "Negeri Johor Darul Takzim (Series 3)"),
    "KA": ("Kedah",          "Negeri Kedah Darul Aman (Series 2)"),
    "NA": ("Negeri Sembilan","Negeri Sembilan Darul Khusus (Series 2)"),
    "PA": ("Pulau Pinang",   "Negeri Pulau Pinang (Series 2)"),
    "PB": ("Pulau Pinang",   "Negeri Pulau Pinang (Series 3)"),
    "TA": ("Terengganu",     "Negeri Terengganu Darul Iman (Series 2)"),
    "QA": ("Sabah",          "Negeri Sabah (Series 2)"),
    "QB": ("Sabah",          "Negeri Sabah (Series 3)"),
    "SA": ("Sarawak",        "Negeri Sarawak (Series 2)"),
    "SB": ("Sarawak",        "Negeri Sarawak (Series 3)"),
    "WA": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 3)"),
    "WB": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 4)"),
    "WC": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 5)"),
    "WD": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 6)"),
    "WG": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 7)"),
    "WK": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 8)"),
    "WL": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 9)"),
    "WM": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 10)"),
    "WP": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 11)"),
    "WQ": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 12)"),
    "WR": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 13)"),
    "WS": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 14)"),
    "WT": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 15)"),
    "WU": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 16)"),
    "WV": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 17)"),
    "WX": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 18)"),
    "WY": ("Kuala Lumpur",   "Wilayah Persekutuan Kuala Lumpur (Series 19)"),

    # Special
    "ATM": ("Military",    "Angkatan Tentera Malaysia"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Uppercase, strip whitespace, collapse internal spaces."""
    return " ".join(text.upper().split())


def _parse_two_row(text: str) -> str:
    """
    Two-row motorcycle plates arrive from EasyOCR as two separate lines.
    Join them into a single string so prefix extraction works uniformly.
    Example: "T 1234\nAB" -> "T 1234 AB"
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_prefix(text: str) -> str | None:
    """
    Extract the leading letter(s) prefix from a normalised plate string.
    Tries two-letter prefix first, then single letter.
    Returns None if no alphabetic prefix found.
    """
    # Match leading alpha chars (1-3 letters) before digits
    match = re.match(r"^([A-Z]{1,3})", text.replace(" ", ""))
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Public interface (required by group contract)
# ---------------------------------------------------------------------------

def classify(plate_text: str, confidence: float = 1.0) -> dict | None:
    """
    Classify a Malaysian vehicle plate text to its registered state.

    Args:
        plate_text: Raw string from EasyOCR (any case, may contain newlines
                    for two-row motorcycle plates).
        confidence: OCR confidence score passed through from ocr.py.

    Returns:
        dict with keys:
            plate_text   - cleaned plate string
            plate_type   - "Standard" | "Two-Row" | "Military" | "Unknown"
            state        - short state name e.g. "Selangor"
            full_state   - full official state name
            prefix       - extracted prefix e.g. "B"
            confidence   - float from OCR
        Returns None if plate_text is empty or prefix cannot be extracted.
    """
    if not plate_text or not plate_text.strip():
        return None

    # Detect and flatten two-row format
    is_two_row = "\n" in plate_text.strip()
    flat_text = _parse_two_row(plate_text) if is_two_row else plate_text
    clean_text = _normalise(flat_text)

    # Detect Diplomatic / Special Plates 
    # Checks for patterns like "12 34 DC" or "123 45 CC"
    diplomatic_match = re.search(r'\d+[\s-]*\d*[\s-]*(DC|CC|UN|PA|WA)$', clean_text)
    if diplomatic_match:
        suffix = diplomatic_match.group(1)
        dip_type = {
            "DC": "Diplomatic Corps",
            "CC": "Consular Corps",
            "UN": "United Nations",
            "PA": "Other International Organisations",
            "WA": "Wakil Asing (Foreign Representative)"
        }
        return {
            "plate_text":  clean_text,
            "plate_type":  "Diplomatic/Special",
            "state":       "International",
            "full_state":  dip_type.get(suffix, "Special Entity"),
            "prefix":      suffix,
            "confidence":  confidence,
        }

    prefix = _extract_prefix(clean_text)
    if prefix is None:
        return None

    # Try longest prefix first (2-letter), then fall back to 1-letter
    state_info = (
        PREFIX_MAP.get(prefix[:3])
        or PREFIX_MAP.get(prefix[:2])
        or PREFIX_MAP.get(prefix[:1])
    )

    if state_info is None:
        return {
            "plate_text":  clean_text,
            "plate_type":  "Unknown",
            "state":       "Unknown",
            "full_state":  "Unrecognised prefix",
            "prefix":      prefix,
            "confidence":  confidence,
        }

    state_name, full_state = state_info

    if prefix == "ATM":
        plate_type = "Military"
    elif is_two_row:
        plate_type = "Square / Two-Row"
    else:
        plate_type = "Standard"

    return {
        "plate_text":  clean_text,
        "plate_type":  plate_type,
        "state":       state_name,
        "full_state":  full_state,
        "prefix":      prefix,
        "confidence":  confidence,
    }