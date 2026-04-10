"""Reusable country-aware field validators and normalization utilities."""

import re


def normalize(value):
    """Standardize text for consistent comparison."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def normalize_nationality(value, country):
    """
    Convert nationality variants into canonical form.
    """
    value = normalize(value)
    country = normalize(country)

    mapping = {
        "ALBANIA": ["ALBANIA", "ALB", "ALBANIAN"],
        "LATVIA": ["LATVIA", "LVA", "LATVIJAS", "LATVIAN"],
        "SLOVAKIA": ["SVK", "SLOVAKIA", "SLOVAK", "SLOVAKIAN"],
    }

    canonical = {
        "ALBANIA": "ALBANIAN",
        "LATVIA": "LATVIAN",
        "SLOVAKIA": "SLOVAK",
    }

    for c, variants in mapping.items():
        if value in variants:
            return canonical[c]

    return value


def is_valid_nationality(value, country):
    """Validate nationality against supported country-specific accepted values."""
    value = normalize_nationality(value, country)
    country = normalize(country)

    rules = {
        "ALBANIA": ["ALBANIAN"],
        "LATVIA": ["LATVIAN"],
        "SLOVAKIA": ["SLOVAK"],
    }

    return value in rules.get(country, [])


def is_valid_personal_no(value, country):
    """Validate personal number against country-specific patterns."""
    value = normalize(value)
    country = normalize(country)

    patterns = {
        "ALBANIA": r"[A-Z]\d{8}[A-Z]",
        "LATVIA": r"\d{6}-\d{5}",
        "SLOVAKIA": r"\d{6}/\d{4}",
    }

    pattern = patterns.get(country)
    return bool(re.fullmatch(pattern, value)) if pattern else False


def is_valid_document_no(value, country):
    """Validate ID/passport number against country-specific patterns."""
    value = normalize(value)
    country = normalize(country)

    patterns = {
        "ALBANIA": r"\d+",
        "LATVIA": r"[A-Z]{2}\d{7}",
        "SLOVAKIA": r"[A-Z]{2}\d{6}",
    }

    pattern = patterns.get(country)
    return bool(re.fullmatch(pattern, value)) if pattern else False


def is_valid_sex(value):
    """Validate sex marker."""
    return normalize(value) in ["M", "F"]
