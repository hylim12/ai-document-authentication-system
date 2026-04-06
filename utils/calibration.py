"""Utilities for post-NER semantic calibration and AML risk scoring."""

from copy import deepcopy


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


def calibrate_entities(entities, raw_lines=None):
    """
    Post-process NER outputs to fix misaligned or invalid fields.

    Supports both:
    - {"FIELD": "value"}
    - {"FIELD": {"text": "value", ...}}
    """
    corrected = deepcopy(entities)

    # RULE 1: GIVEN NAME should not contain label artifacts
    given_name = _entity_text(corrected.get("GIVEN NAME"))
    if given_name:
        val = str(given_name).lower()
        if "surname" in val or "name" in val:
            corrected.pop("GIVEN NAME", None)

    # RULE 2: PERSONAL NO must be numeric
    personal_no = _entity_text(corrected.get("PERSONAL NO"))
    if personal_no and not str(personal_no).isdigit():
        corrected.pop("PERSONAL NO", None)

    # RULE 3: SURNAME must be alphabetic
    surname = _entity_text(corrected.get("SURNAME"))
    if surname and not str(surname).isalpha():
        corrected.pop("SURNAME", None)

    # RULE 4: Recover GIVEN NAME using SURNAME context
    if not corrected.get("GIVEN NAME") and raw_lines:
        surname = _entity_text(corrected.get("SURNAME"))
        if surname:
            surname_lower = str(surname).lower()
            for line in raw_lines:
                line_text = str(line).strip()
                if not line_text:
                    continue
                if surname_lower in line_text.lower():
                    parts = line_text.split()
                    if len(parts) >= 2:
                        _set_entity_text(corrected, "GIVEN NAME", parts[0])
                        break

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


def compute_risk_score(entities):
    """
    Assign AML risk score based on missing/invalid fields.
    """
    risk_score = 0
    issues = []

    critical_fields = ["GIVEN NAME", "SURNAME", "NATIONALITY"]

    # Missing fields
    for field in critical_fields:
        if not _entity_text(entities.get(field)):
            risk_score += 2
            issues.append(f"{field} missing")

    # Invalid GIVEN NAME
    given_name = _entity_text(entities.get("GIVEN NAME"))
    if given_name and not str(given_name).isalpha():
        risk_score += 1
        issues.append("Invalid GIVEN NAME")

    # Nationality mismatch check
    pob = str(_entity_text(entities.get("PLACE OF BIRTH")) or "")
    nationality = str(_entity_text(entities.get("NATIONALITY")) or "")
    if "ALB" in pob.upper() and nationality.lower() != "albanian":
        risk_score += 3
        issues.append("Nationality mismatch")

    return risk_score, issues
