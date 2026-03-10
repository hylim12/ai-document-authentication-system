"""LLM-based NER extraction for passport/ID OCR JSON outputs."""

import json
from typing import Dict, List, Any

from prompts.passport_ner_prompt import build_passport_ner_prompt


def load_ocr_json(json_path: str) -> List[Dict[str, Any]]:
    """Load OCR JSON file and return normalized OCR rows for prompting."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("ocr_results", [])
    normalized = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        bbox = row.get("bbox", None)
        conf = row.get("confidence", 0.0)
        if not text:
            continue
        normalized.append({
            "text": text,
            "bbox": bbox,
            "confidence": float(conf) if conf is not None else 0.0,
        })
    return normalized


def _safe_parse_json(content: str) -> Dict[str, Any]:
    """Parse JSON content; fallback to extracting the largest JSON object."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise


def extract_passport_fields_llm(ocr_json_path: str, model: str = "gpt-4o-mini") -> Dict[str, Dict[str, Any]]:
    """Extract dynamic passport/ID entities from OCR JSON using OpenAI API."""
    # Keep OpenAI import local so the rest of the project can run without SDK installed.
    from openai import OpenAI

    ocr_rows = load_ocr_json(ocr_json_path)
    prompt = build_passport_ner_prompt(ocr_rows)

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict JSON information extraction engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content or "{}"
    payload = _safe_parse_json(content)
    entities = payload.get("entities", [])

    normalized_entities: Dict[str, Dict[str, Any]] = {}
    for item in entities:
        field = str(item.get("field", "")).strip().upper()
        text = str(item.get("text", "")).strip()
        if not field or not text:
            continue

        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = None

        confidence = item.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        normalized_entities[field] = {
            "text": text,
            "bbox": tuple(bbox) if bbox else None,
            "confidence": confidence,
        }

    return normalized_entities