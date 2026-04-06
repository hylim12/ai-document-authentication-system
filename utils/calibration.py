"""Utilities for post-NER semantic calibration and AML risk scoring."""

from copy import deepcopy
import re


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


def validate_personal_no(value, country):
    """Validate PERSONAL NO using country-specific formats."""
    value = re.sub(r'[^A-Z0-9/-]', '', str(value or '').upper())

    if country == "ALBANIA":
        return bool(re.fullmatch(r'[A-Z]\d{8}[A-Z]', value))

    elif country == "LATVIA":
        return bool(re.fullmatch(r'\d{6}-\d{5}', value))

    elif country == "SLOVAKIA":
        return bool(re.fullmatch(r'\d{6}/\d{4}', value))

    return False


def validate_id_card(value, country):
    """Validate ID/PASSPORT number using country-specific formats."""
    value = re.sub(r'[^A-Z0-9]', '', str(value or '').upper())

    if country == "ALBANIA":
        return bool(re.fullmatch(r'\d+', value))

    elif country == "LATVIA":
        return bool(re.fullmatch(r'[A-Z]{2}\d{7}', value))

    elif country == "SLOVAKIA":
        return bool(re.fullmatch(r'[A-Z]{2}\d{6}', value))

    return False


def clean_personal_no(value):
    """Clean OCR noise while preserving valid format characters."""
    if not value:
        return None

    value = str(value).upper().strip()
    value = re.sub(r"[^A-Z0-9/-]", "", value)
    return value or None


def enforce_nationality(entities, country):
    """Strictly enforce expected nationality for supported countries."""
    nat = _entity_text(entities.get("NATIONALITY"))

    expected = {
        "ALBANIA": "ALBANIAN",
        "LATVIA": "LATVIJAS",
        "SLOVAKIA": "SVK"
    }

    if nat:
        if nat.upper() != expected.get(country, ""):
            entities["NATIONALITY"] = None

    return entities


def calibrate_entities(entities, country=None, raw_lines=None):
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
        if pno and not validate_personal_no(pno, country):
            corrected.pop("PERSONAL NO", None)

        id_val = _entity_text(corrected.get("ID CARD NO")) or _entity_text(corrected.get("PASSPORT NO"))
        if id_val and not validate_id_card(id_val, country):
            corrected.pop("ID CARD NO", None)
            corrected.pop("PASSPORT NO", None)

        corrected = enforce_nationality(corrected, country)

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

    if country == "ALBANIA":
        if get("NATIONALITY") != "ALBANIAN":
            risk += 4
            issues.append("Forged nationality (ALBANIA)")

        if not validate_personal_no(get("PERSONAL NO"), country):
            risk += 3
            issues.append("Invalid Personal No (ALBANIA)")

        if not validate_id_card(get("ID CARD NO"), country):
            risk += 3
            issues.append("Invalid ID Card No (ALBANIA)")

        pob = get("PLACE OF BIRTH") or ""
        if "ALB" not in pob.upper():
            risk += 2
            issues.append("Invalid Place of Birth (ALBANIA)")

    elif country == "LATVIA":
        if get("NATIONALITY") != "LATVIJAS":
            risk += 4
            issues.append("Forged nationality (LATVIA)")

        if not validate_id_card(get("PASSPORT NO"), country):
            risk += 3
            issues.append("Invalid Passport No")

        if not validate_personal_no(get("PERSONAL NO"), country):
            risk += 3
            issues.append("Invalid Personal No")

        if get("SEX") not in ["M", "F"]:
            risk += 1
            issues.append("Invalid sex")

    elif country == "SLOVAKIA":
        if get("NATIONALITY") != "SVK":
            risk += 4
            issues.append("Forged nationality (SLOVAKIA)")

        if not validate_id_card(get("ID CARD NO"), country):
            risk += 3
            issues.append("Invalid ID Card No")

        if not validate_personal_no(get("PERSONAL NO"), country):
            risk += 3
            issues.append("Invalid Personal No")

    return risk, issues
