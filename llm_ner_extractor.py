"""LLM-based NER extraction for passport/ID OCR JSON outputs."""

import json
import os
from typing import Dict, List, Any

from prompts.passport_ner_prompt import build_passport_ner_prompt



class LLMNERQuotaError(RuntimeError):
    """Raised when API quota/billing limits block LLM NER."""


class LLMNERConfigError(RuntimeError):
    """Raised when LLM NER is not correctly configured."""


def _load_key_from_dotenv(dotenv_path: str = ".env") -> str:
    """Best-effort local .env loader for OPENROUTER_API_KEY."""
    if not os.path.exists(dotenv_path):
        return ""
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    return v.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _resolve_llm_config(default_model: str) -> tuple[str, str, str]:
    """Resolve API key, base URL and model, preferring OpenRouter settings."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        api_key = _load_key_from_dotenv()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.getenv("OPENROUTER_MODEL", default_model).strip() or default_model
    return api_key, base_url, model




def _canonicalize_field_name(field: str) -> str:
    """Normalize field names from LLM output to detector's internal canonical keys."""
    norm = " ".join(str(field).upper().replace("_", " ").replace("-", " ").split())
    aliases = {
        "GIVEN NAME": "GIVEN NAME",
        "GIVEN NAMES": "GIVEN NAME",
        "GENDER": "SEX",
        "DATE OF BIRTH": "DATE OF BIRTH",
        "DATE OF ISSUE": "DATE OF ISSUE",
        "DATE OF EXPIRY": "DATE OF EXPIRY",
        "PLACE OF BIRTH": "PLACE OF BIRTH",
        "ID CARD NUMBER": "ID CARD NO",
        "ID NUMBER": "ID CARD NO",
        "CARD NO": "ID CARD NO",
        "PASSPORT NUMBER": "PASSPORT NO",
        "PERSONAL NUMBER": "PERSONAL NO",
        "MRZ LINE 1": "MRZ LINE 1",
        "MRZ LINE 2": "MRZ LINE 2",
        "FULL NAME": "FULL NAME",
        "HEIGHT": "HEIGHT",
        "SIGNATURE": "SIGNATURE",
        "AUTHORITY": "AUTHORITY",
        "NATIONALITY": "NATIONALITY",
        "SURNAME": "SURNAME",
        "SEX": "SEX",
        "PASSPORT NO": "PASSPORT NO",
        "ID CARD NO": "ID CARD NO",
        "PERSONAL NO": "PERSONAL NO",
    }
    return aliases.get(norm, norm)


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


def extract_passport_fields_llm(ocr_json_path: str, model: str = "openai/gpt-4o-mini") -> Dict[str, Dict[str, Any]]:
    """Extract dynamic passport/ID entities from OCR JSON using OpenRouter (OpenAI SDK)."""
    api_key, base_url, resolved_model = _resolve_llm_config(model)
    if not api_key:
        raise LLMNERConfigError(
            "OPENROUTER_API_KEY is not set (or .env missing). Configure it to enable LLM NER."
        )

    # Keep OpenAI import local so the rest of the project can run without SDK installed.
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMNERConfigError(
            "openai package is not installed. Install it to enable LLM NER."
        ) from e

    ocr_rows = load_ocr_json(ocr_json_path)
    prompt = build_passport_ner_prompt(ocr_rows)

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": "You are a strict JSON information extraction engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "Error code: 429" in msg or "quota" in msg.lower():
            raise LLMNERQuotaError(
                "LLM provider quota exceeded (429 insufficient_quota). "
                "Check billing/plan, or continue with regex fallback."
            ) from e
        raise

    content = response.choices[0].message.content or "{}"
    payload = _safe_parse_json(content)
    entities = payload.get("entities", [])

    normalized_entities: Dict[str, Dict[str, Any]] = {}
    for item in entities:
        field = _canonicalize_field_name(item.get("field", ""))
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