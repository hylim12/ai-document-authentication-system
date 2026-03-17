"""Prompt templates for LLM-based passport/ID NER extraction."""

import json


def build_passport_ner_prompt(ocr_results):
    """Build a strict JSON extraction prompt from OCR results."""
    ocr_payload = json.dumps(ocr_results, ensure_ascii=False, indent=2)

    return f"""
You are an expert AI system for extracting structured named entities from passports and national ID cards for AML/KYC verification and document forgery detection.

--------------------------------------------------
TASK
--------------------------------------------------
You will receive an OCR payload containing text blocks extracted from a scanned identity document.
Each block includes:
  - "text": the recognized string
  - "bbox": bounding box coordinates [x1, y1, x2, y2]

Your task is to:
1. Identify the document type and issuing country.
2. Match field labels to their corresponding field values using label text, spatial proximity, and document structure patterns.
3. Extract and return all identifiable fields in strict JSON format.

You must NOT rely on fixed country-specific templates.
You must NOT invent or hallucinate values.
You must preserve extracted text exactly as it appears in the OCR output.

--------------------------------------------------
SUPPORTED DOCUMENT TYPES
--------------------------------------------------
- Passport (TYPE_P)
- National ID Card (TYPE_ID)

Document languages may include English, Albanian, Latvian, Slovak, and other European languages.
Field labels may appear in one or more languages on the same document (e.g., "date of birth / datëlindja").

--------------------------------------------------
FIELD LABEL DICTIONARY
--------------------------------------------------
Use the mappings below to recognize field labels across languages.
Labels may appear in any combination, order, or capitalization.

DOCUMENT_TYPE:
  - type, tips, type/type

ISSUING_COUNTRY:
  - code of issuing state, valsts kods, code du pays

PASSPORT_NO:
  - passport no, passport no., pases nr., passeport no, nr. letërnjoftim, card no

ID_CARD_NO:
  - číslo, no., number, card no, nr. letërnjoftim

SURNAME:
  - surname, nom, mbiemri, uzvārds, priezvisko

GIVEN_NAME:
  - given name, given names, prénom(s), emri, vārds(-i), meno

NATIONALITY:
  - nationality, nationalité, shtetësia, pilsonība, štátne občianstvo

DATE_OF_BIRTH:
  - date of birth, date de naissance, datëlindja, dzimšanas datums, dátum narodenia

PLACE_OF_BIRTH:
  - place of birth, lieu de naissance, vendlindja, dzimšanas vieta

SEX:
  - sex, sexe, gjinia, dzimums, pohlavie
  - Note: value may appear as combined male indicators such as "M", "V/M", or "M/V" — always normalize to "M" or "F"

HEIGHT:
  - height, taille, augums

PERSONAL_NO:
  - personal no, personal no., nr. personal, personas kods, rodné číslo, code d'identité

DATE_OF_ISSUE:
  - date of issue, date de délivrance, data e lëshimit, izdošanas datums, dátum vydania

DATE_OF_EXPIRY:
  - date of expiry, date d'expiration, data e skadimit, derīguma termiņš, dátum platnosti

AUTHORITY:
  - authority, autorité, autoriteti lëshues, izdevējiestāde, vydal, issued by

SIGNATURE:
  - signature, firma, paraksts, podpis držiteľa

--------------------------------------------------
FIELDS TO EXTRACT (WHEN PRESENT)
--------------------------------------------------
Always attempt to extract the following canonical fields if identifiable:

  DOCUMENT_TYPE         - "PASSPORT" or "ID_CARD"
  ISSUING_COUNTRY       - ISO 3-letter country code (e.g., ALB, LVA, SVK)
  SURNAME
  GIVEN_NAME
  FULL_NAME             - Concatenation of GIVEN_NAME + SURNAME if no separate full name field exists
  PASSPORT_NO           - For passports
  ID_CARD_NO            - For national ID cards
  PERSONAL_NO           - National personal identifier number (may include slashes or dashes, preserve as-is)
  NATIONALITY
  PLACE_OF_BIRTH
  DATE_OF_BIRTH
  DATE_OF_ISSUE
  DATE_OF_EXPIRY
  SEX                   - Normalize to "M" or "F"
  HEIGHT                - Numeric value only (e.g., "175")
  AUTHORITY
  MRZ_LINE_1            - First MRZ line (raw, preserve all < characters)
  MRZ_LINE_2            - Second MRZ line (raw, preserve all < characters)

--------------------------------------------------
SPATIAL REASONING RULES
--------------------------------------------------
Use these rules to associate labels with values when OCR blocks are separate:

1. PROXIMITY: A value block is most likely directly to the right of, or immediately below, its label block.
2. ALIGNMENT: Values in the same horizontal band as a label belong to that label.
3. MULTILINGUAL LABELS: When a label appears in two languages (e.g., "surname / mbiemri"), treat both as the same field label.
4. INLINE VALUES: Some blocks contain both the label and value in the same text string (e.g., "Card No. 367253746"). Split and extract accordingly.
5. STACKED LABELS: When a label appears above another label with no intervening value, the value below the second label belongs to the second label.
6. MRZ ZONE: MRZ lines appear at the bottom of the document in monospace font and always begin with a document type prefix (e.g., "P<" for passports). Extract MRZ lines verbatim including all filler characters (<).

--------------------------------------------------
CONFIDENCE SCORING GUIDE
--------------------------------------------------
Assign a confidence score between 0.0 and 1.0 per field:

  1.0  - Field label explicitly present and value clearly OCR'd
  0.9  - Field label present, value inferred from strong spatial proximity
  0.8  - Field label present in another language, value confidently matched
  0.7  - Value inferred from MRZ or cross-validated with another field
  0.5  - Ambiguous label or partially OCR'd value
  0.3  - Value guessed from document structure with no clear label
  0.0  - Field not found or could not be extracted

Only return fields with a confidence score of 0.3 or higher.

--------------------------------------------------
MRZ EXTRACTION AND CROSS-VALIDATION
--------------------------------------------------
If MRZ lines are present:
- Extract both lines verbatim as MRZ_LINE_1 and MRZ_LINE_2.
- Use MRZ data to cross-validate or supplement the following visual fields:
    PASSPORT_NO / ID_CARD_NO  → characters 1–9 of MRZ line 2
    NATIONALITY               → characters 10–12 of MRZ line 2
    DATE_OF_BIRTH             → characters 14–19 of MRZ line 2 (YYMMDD)
    DATE_OF_EXPIRY            → characters 21–26 of MRZ line 2 (YYMMDD)
    SEX                       → character 20 of MRZ line 2
    PERSONAL_NO               → characters 29–42 of MRZ line 2 (if present)
    SURNAME / GIVEN_NAME      → from MRZ line 1, after country code, split on "<<"

If a visual field and MRZ-derived field conflict, return both candidates and lower the confidence score to 0.6 or below.

--------------------------------------------------
EDGE CASE HANDLING
--------------------------------------------------
1. COMBINED SEX VALUES: "V/M" or "M/V" in Latvian documents — extract only "M" or "F" based on known male/female indicators per language.
2. PERSONAL_NO WITH PUNCTUATION: Preserve slashes and dashes (e.g., "550522/3941", "280974-14045").
3. DATE FORMATS: Preserve dates exactly as shown (e.g., "28.09.1974", "20-11-1991", "13-04-2029").
4. DUAL-PHOTO DOCUMENTS: Some ID cards contain two photos (main + small). Do not extract photo regions as fields.
5. AUTHORITY MULTI-LINE: Authority may span multiple OCR lines (e.g., "PMLP CĒSU" + "NODAĻA"). Concatenate with a space.
6. ISSUING_COUNTRY FROM HEADER: If not a separate label, infer from document header text (e.g., "REPUBLIC OF ALBANIA" → ALB) or MRZ country code.
7. DOCUMENT_TYPE FROM HEADER: Infer from header keywords (e.g., "PASSPORT / PASE" → PASSPORT, "ID-CARD / LETËRNJOFTIM" → ID_CARD).

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON — NO OTHER TEXT)
--------------------------------------------------
Return ONLY a valid JSON object. Do not include markdown, backticks, explanations, or any text outside the JSON.

{{
  "document_type": "PASSPORT" | "ID_CARD",
  "issuing_country": "<ISO 3-letter code>",
  "entities": [
    {{
      "field": "<CANONICAL_FIELD_NAME>",
      "text": "<extracted_value_as_seen_in_OCR>",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }}
  ]
}}

Rules:
- "field" must use canonical names defined in FIELDS TO EXTRACT.
- "text" must be the raw OCR string, not reformatted.
- "bbox" should be the bounding box of the VALUE block (not the label). If undetermined, use null.
- "confidence" must be a float between 0.0 and 1.0.
- If a field is found multiple times, return the single best candidate only.
- Do not return fields with confidence below 0.3.

--------------------------------------------------
OCR INPUT PAYLOAD
--------------------------------------------------
{ocr_payload}

Return JSON only.
""".strip()