"""Prompt templates for LLM-based passport NER extraction."""

import json


def build_passport_ner_prompt(ocr_results):
    """Build a strict JSON extraction prompt from OCR results."""
    ocr_payload = json.dumps(ocr_results, ensure_ascii=False, indent=2)
    return f"""
You are an AML passport/ID field extraction system.

Task:
- Read OCR data and identify document fields and values dynamically.
- Do not rely on fixed country-specific templates.
- Return only fields you can infer with reasonable confidence.

Output format (STRICT JSON only, no markdown):
{{
  "entities": [
    {{
      "field": "<canonical field name>",
      "text": "<extracted value>",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Use canonical English field names when possible (e.g., SURNAME, GIVEN NAME, DATE OF BIRTH, DATE OF ISSUE, DATE OF EXPIRY, SEX, NATIONALITY, PLACE OF BIRTH, PASSPORT NO, ID CARD NO, PERSONAL NO, AUTHORITY, SIGNATURE).
- If a field appears multiple times, keep the best candidate.
- bbox should come from OCR evidence; if unavailable use null.
- confidence must be 0..1.
- Return valid JSON only.

OCR DATA:
{ocr_payload}
""".strip()