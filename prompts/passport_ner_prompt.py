"""Prompt templates for LLM-based passport/ID NER extraction."""

import json


def build_passport_ner_prompt(ocr_results, regex_entities=None):
    """Build a strict JSON extraction prompt from OCR results and optional regex baseline entities."""
    ocr_payload = json.dumps(ocr_results, ensure_ascii=False, indent=2)
    regex_payload = json.dumps(regex_entities or {}, ensure_ascii=False, indent=2)

    return f"""
You are an expert AI system for extracting structured data from passports and national ID cards used in AML/KYC verification.

You are given:
1. OCR results
2. Pre-extracted fields from regex (may be incomplete or slightly incorrect)

Task:
Analyze OCR text blocks and identify passport or ID fields and their corresponding values.
Documents may come from different countries and languages.

Your task:
- Validate and correct pre-extracted fields
- Fill missing fields
- Improve confidence
- Do NOT remove correct fields
- You MUST extract ALL fields listed when present in OCR

The OCR results contain:
- text
- bounding boxes (bbox)

You must infer fields using:
1. Field labels (e.g., surname, nationality)
2. Spatial proximity between labels and values
3. Document structure patterns
4. MRZ (Machine Readable Zone) if present

Do NOT rely on fixed country templates.

--------------------------------------------------
DOCUMENT TYPES
--------------------------------------------------

The document may be:
- Passport
- National ID card

Languages may include:
- English
- Albanian
- Latvian
- Slovak
- Other European languages

Field labels may appear in multiple languages.

Example label mappings:

Surname:
- surname
- mbiemri
- uzvārds
- priezvisko

Given Name:
- given name
- emri
- vārds
- meno

Nationality:
- nationality
- shtetësia
- pilsonība
- štátne občianstvo

Date of Birth:
- date of birth
- datëlindja
- dzimšanas datums
- dátum narodenia

Sex:
- sex
- gjinia
- dzimums
- pohlavie

Date of Issue:
- date of issue
- data e lëshimit
- izdošanas datums
- dátum vydania

Date of Expiry:
- date of expiry
- data e skadimit
- derīguma termiņš
- dátum platnosti

Place of Birth:
- vendlindja
- place of birth
- dzimšanas vieta

Authority:
- authority
- autoriteti
- issuing authority
- iestāde

Personal Number:
- personal no
- personas kods
- rodné číslo

Passport Number / ID Number:
- passport no
- card no
- number
- nr
- číslo

--------------------------------------------------
FIELDS TO EXTRACT (WHEN PRESENT)
--------------------------------------------------

Extract fields if identifiable:

SURNAME
GIVEN_NAME
FULL_NAME
PASSPORT_NO
ID_CARD_NO
PERSONAL_NO
NATIONALITY
PLACE_OF_BIRTH
DATE_OF_BIRTH
DATE_OF_ISSUE
DATE_OF_EXPIRY
SEX
HEIGHT
AUTHORITY
SIGNATURE
MRZ_LINE_1
MRZ_LINE_2

If multiple candidates exist, choose the most reliable value.

--------------------------------------------------
MRZ EXTRACTION
--------------------------------------------------

If the document contains Machine Readable Zone lines:

Example:

P<LVAALKSNIS<<AINARS<<<<<<
LV6309038LVA7409288M2611044

Extract them as:

MRZ_LINE_1
MRZ_LINE_2

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------------------------

Return ONLY valid JSON.

{{
  "entities": [
    {{
      "field": "<canonical_field_name>",
      "text": "<extracted_value>",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }}
  ]
}}

--------------------------------------------------
RULES
--------------------------------------------------

1. Use canonical English field names.
2. Only return fields with reasonable confidence.
3. Do not invent values.
4. If bbox cannot be determined, return null.
5. Confidence must be between 0 and 1.
6. If a field appears multiple times, return the best candidate.
7. Preserve text exactly as shown in OCR.

--------------------------------------------------
FEW-SHOT EXAMPLE
--------------------------------------------------

Example Input OCR:
[
  {{"text": "Surname", "bbox": [10, 10, 100, 30], "confidence": 0.99}},
  {{"text": "Agani", "bbox": [120, 10, 220, 30], "confidence": 0.98}}
]

Example Output:
{{
  "entities": [
    {{"field": "SURNAME", "text": "Agani", "bbox": [120, 10, 220, 30], "confidence": 0.98}}
  ]
}}

--------------------------------------------------
PRE-EXTRACTED FIELDS (REGEX BASELINE)
--------------------------------------------------

{regex_payload}

--------------------------------------------------
OCR DATA
--------------------------------------------------

{ocr_payload}

Return JSON only.
""".strip()