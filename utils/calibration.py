"""Utilities for post-field semantic calibration and AML risk scoring."""

from copy import deepcopy
import re

from utils.validators import (
    is_valid_document_no,
    is_valid_nationality,
    is_valid_personal_no,
    is_valid_sex,
    normalize_nationality,
    normalize,
)


def _entity_text(value):
    """Return entity text regardless of whether value is dict or string."""
    if isinstance(value, dict):
        return value.get("text")
    return value


def _set_entity_text(entities, field, new_text):
    """Set entity text while preserving existing payload shape."""
    current = entities.get(field)
    if isinstance(current, dict):
        payload = current.copy()
        payload["text"] = new_text
        entities[field] = payload
    else:
        entities[field] = {
            "text": new_text,
            "bbox": (0, 0, 0, 0),
            "confidence": 0.5,
        }


def clean_personal_no(value):
    """Clean OCR noise while preserving valid format characters."""
    if not value:
        return None

    value = normalize(value)
    value = re.sub(r"[^A-Z0-9/-]", "", value)
    return value or None


def calibrate_entities(entities, country=None, raw_lines=None):
    """
    Post-process Field outputs to fix misaligned or invalid fields.

    Supports both:
    - {"FIELD": "value"}
    - {"FIELD": {"text": "value", ...}}
    """
    corrected = deepcopy(entities)

    if country == "SLOVAKIA":
        corrected.pop("PLACE OF BIRTH", None)

    # RULE 1: GIVEN NAME should not contain label artifacts
    given_name = _entity_text(corrected.get("GIVEN NAME"))
    if given_name:
        val = str(given_name).lower()
        if "surname" in val or "name" in val:
            corrected.pop("GIVEN NAME", None)

    # RULE 2: Basic cleanup for PERSONAL NO
    personal_no = _entity_text(corrected.get("PERSONAL NO"))
    if personal_no:
        cleaned = clean_personal_no(personal_no)
        _set_entity_text(corrected, "PERSONAL NO", cleaned)

    # RULE 3: SURNAME must be alphabetic
    surname = _entity_text(corrected.get("SURNAME"))
    if surname and not str(surname).isalpha():
        corrected.pop("SURNAME", None)

    # RULE 4: Recover PERSONAL NO from OCR lines when missing
    if not _entity_text(corrected.get("PERSONAL NO")) and raw_lines:
        for line in raw_lines:
            matches = re.findall(r"[A-Z]\d{6,}", str(line).upper())
            if matches:
                _set_entity_text(corrected, "PERSONAL NO", matches[0])
                break

    # RULE 5: Recover GIVEN NAME using SURNAME context
    if not corrected.get("GIVEN NAME") and raw_lines:
        surname = _entity_text(corrected.get("SURNAME"))
        for line in raw_lines:
            words = re.findall(r"[A-Za-z]{3,}", str(line))

            if len(words) >= 2:
                for w in words:
                    if w.lower() != str(surname).lower():
                        _set_entity_text(corrected, "GIVEN NAME", w)
                        break
            if corrected.get("GIVEN NAME"):
                break

    # RULE 6: Keep PERSONAL NO distinct from ID CARD NO when comparable
    personal_no = _entity_text(corrected.get("PERSONAL NO"))
    id_card_no = _entity_text(corrected.get("ID CARD NO"))
    if personal_no and id_card_no:
        personal_no = str(personal_no)
        id_card_no = str(id_card_no)
        if len(personal_no) > 1 and personal_no[1:] == id_card_no:
            # likely same number -> valid case, keep both
            pass

    # COUNTRY-AWARE VALIDATION
    if country:
        pno = _entity_text(corrected.get("PERSONAL NO"))
        if pno and not is_valid_personal_no(pno, country):
            corrected.pop("PERSONAL NO", None)

        id_val = _entity_text(corrected.get("ID CARD NO")) or _entity_text(corrected.get("PASSPORT NO"))
        if id_val and not is_valid_document_no(id_val, country):
            corrected.pop("ID CARD NO", None)
            corrected.pop("PASSPORT NO", None)

        nat = _entity_text(corrected.get("NATIONALITY"))
        if nat:
            normalized_nat = normalize_nationality(nat, country)
            _set_entity_text(corrected, "NATIONALITY", normalized_nat)

            if not is_valid_nationality(normalized_nat, country):
                corrected.pop("NATIONALITY", None)

    return corrected


def derive_nationality(entities):
    """
    Infer nationality from PLACE OF BIRTH when missing.
    """
    enriched = deepcopy(entities)
    nationality = _entity_text(enriched.get("NATIONALITY"))
    if not nationality:
        pob = str(_entity_text(enriched.get("PLACE OF BIRTH")) or "")
        if "ALB" in pob.upper():
            _set_entity_text(enriched, "NATIONALITY", "Albanian")
        else:
            _set_entity_text(enriched, "NATIONALITY", "UNKNOWN")

    return enriched


def compute_risk_score(entities, country=None):
    """Assign AML risk score based on global and country-specific rules."""
    risk = 0
    issues = []

    def get(field):
        return _entity_text(entities.get(field))

    for field in ["GIVEN NAME", "SURNAME"]:
        if not get(field):
            risk += 2
            issues.append(f"{field} missing")

    if country != "SLOVAKIA":
        if not get("PLACE OF BIRTH"):
            risk += 2
            issues.append("PLACE OF BIRTH missing")

    if country == "ALBANIA":
        nat = normalize_nationality(get("NATIONALITY"), country)
        if not is_valid_nationality(nat, country):
            risk += 4
            issues.append(f"Forged nationality ({country})")

        if not is_valid_personal_no(get("PERSONAL NO"), country):
            risk += 3
            issues.append("Invalid Personal No")

        if not is_valid_document_no(get("ID CARD NO") or get("PASSPORT NO"), country):
            risk += 3
            issues.append("Invalid Document Number")

    elif country == "LATVIA":
        nat = normalize_nationality(get("NATIONALITY"), country)
        if not is_valid_nationality(nat, country):
            risk += 4
            issues.append(f"Forged nationality ({country})")

        if not is_valid_document_no(get("ID CARD NO") or get("PASSPORT NO"), country):
            risk += 3
            issues.append("Invalid Document Number")

        if not is_valid_personal_no(get("PERSONAL NO"), country):
            risk += 3
            issues.append("Invalid Personal No")

        if not is_valid_sex(get("SEX")):
            risk += 1
            issues.append("Invalid sex")

    elif country == "SLOVAKIA":
        nat = normalize_nationality(get("NATIONALITY"), country)
        if not is_valid_nationality(nat, country):
            risk += 4
            issues.append(f"Forged nationality ({country})")

        if not is_valid_document_no(get("ID CARD NO") or get("PASSPORT NO"), country):
            risk += 3
            issues.append("Invalid Document Number")

        if not is_valid_personal_no(get("PERSONAL NO"), country):
            risk += 3
            issues.append("Invalid Personal No")

    return risk, issues
