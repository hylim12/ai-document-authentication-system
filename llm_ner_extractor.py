"""LLM-based NER extraction for passport/ID OCR JSON outputs."""

import ast
import json
import os
import re
from typing import Dict, List, Any

from prompts.passport_ner_prompt import build_passport_ner_prompt

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

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
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
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

import re

def _filter_ocr_rows_for_llm(rows):
    """
    Reduce OCR rows to only those likely containing identity fields.
    This greatly reduces token usage for the LLM.
    """
    keyword_pattern = re.compile(
        r"(surname|given|name|birth|date|issue|expiry|sex|nationality|place|personal|passport|card|authority|height|mrz|nr|no)",
        re.IGNORECASE
    )

    filtered = []

    for r in rows:
        text = r.get("text", "").strip()

        # keep if contains keyword
        if keyword_pattern.search(text):
            filtered.append(r)
            continue

        # keep if looks like a date
        if re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", text):
            filtered.append(r)
            continue

        # keep if long alphanumeric (passport numbers etc.)
        if re.search(r"[A-Z0-9]{6,}", text):
            filtered.append(r)
            continue

        # keep MRZ-like lines
        if "<" in text:
            filtered.append(r)

    return filtered

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


def _strip_markdown_fences(content: str) -> str:
    """Remove common markdown code fences around JSON payloads."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
    return content.strip()


def _extract_largest_balanced_json_object(content: str) -> str:
    """Extract the largest balanced {...} JSON object while respecting quoted strings."""
    best = ""
    stack = []
    in_string = False
    escape = False
    start_idx = None

    for i, ch in enumerate(content):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == '{':
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == '}' and stack:
            stack.pop()
            if not stack and start_idx is not None:
                candidate = content[start_idx:i + 1]
                if len(candidate) > len(best):
                    best = candidate
                start_idx = None

    return best


def _parse_python_literal_object(candidate: str) -> Dict[str, Any] | None:
    """Parse Python-literal style dict/list output (single quotes, True/False/None)."""
    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _repair_common_json_issues(candidate: str) -> str:
    """Repair minor JSON issues often produced by LLMs."""
    repaired = candidate
    # Remove trailing commas before object/array close.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Replace non-standard numeric literals with null.
    repaired = re.sub(r"\bNaN\b|\bInfinity\b|-Infinity", "null", repaired)
    return repaired


def _safe_parse_json(content: str) -> Dict[str, Any]:
    """Parse JSON content with safe fallbacks for markdown and minor format glitches."""
    content = _strip_markdown_fences(content)

    parse_attempts = [content]
    extracted = _extract_largest_balanced_json_object(content)
    if extracted:
        parse_attempts.append(extracted)
        parse_attempts.append(_repair_common_json_issues(extracted))

    last_error = None
    for attempt in parse_attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as e:
            last_error = e

        python_like = _parse_python_literal_object(attempt)
        if python_like is not None:
            return python_like

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Failed to parse JSON", content, 0)


def _create_ner_completion(client, resolved_model: str, prompt: str):
    """Create completion, preferring JSON-formatted responses when supported."""
    base_messages = [
        {"role": "system", "content": "You are a strict JSON information extraction engine."},
        {"role": "user", "content": prompt},
    ]

    common_kwargs = {
        "model": resolved_model,
        "messages": base_messages,
        "temperature": 0,
        "max_tokens": 200,
        "top_p": 1,
    }

    try:
        return client.chat.completions.create(
            **common_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        msg = str(e).lower()
        unsupported = any(token in msg for token in (
            "response_format",
            "json_schema",
            "not supported",
            "unsupported",
            "invalid_request_error",
        ))
        if unsupported:
            return client.chat.completions.create(**common_kwargs)
        raise

def extract_passport_fields_llm(ocr_json_path: str, model: str = DEFAULT_OPENROUTER_MODEL) -> Dict[str, Dict[str, Any]]:
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
        response = _create_ner_completion(client, resolved_model, prompt)
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