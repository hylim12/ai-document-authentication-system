"""LLM-based NER extraction for passport/ID OCR JSON outputs."""

import ast
import http.client
import json
import os
import re
import time
from typing import Dict, List, Any
from urllib import error
from urllib.parse import urlparse

from prompts.passport_ner_prompt import build_passport_ner_prompt

DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LOCAL_LLM_MODEL = "llama3.2:latest"

class LLMNERQuotaError(RuntimeError):
    """Raised when API quota/billing limits block LLM NER."""

class LLMNERConfigError(RuntimeError):
    """Raised when LLM NER is not correctly configured."""

class LLMNERAuthError(RuntimeError):
    """Raised when provider authentication/authorization blocks LLM NER."""


class LLMNERTokenLimitError(RuntimeError):
    """Raised when the LLM request exceeds model context/token limits."""


class _PersistentHTTPJSONClient:
    """Small keep-alive HTTP client cache for repeated local LLM calls."""

    def __init__(self):
        self._connections: Dict[tuple[str, str, int], http.client.HTTPConnection] = {}

    def post_json(self, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 120) -> Dict[str, Any]:
        parsed = urlparse(endpoint)
        scheme = parsed.scheme or "http"
        host = parsed.hostname
        if not host:
            raise LLMNERConfigError(f"Invalid endpoint URL: {endpoint}")

        if parsed.port is not None:
            port = parsed.port
        elif scheme == "https":
            port = 443
        else:
            port = 80

        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        key = (scheme, host, port)
        conn = self._connections.get(key)
        if conn is None:
            conn = http.client.HTTPSConnection(host, port, timeout=timeout) if scheme == "https" else http.client.HTTPConnection(host, port, timeout=timeout)
            self._connections[key] = conn

        body = json.dumps(payload).encode("utf-8")
        req_headers = {**headers, "Connection": "keep-alive"}

        try:
            conn.request("POST", path, body=body, headers=req_headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
        except Exception:
            conn.close()
            self._connections.pop(key, None)
            raise

        if resp.status >= 400:
            raise error.HTTPError(endpoint, resp.status, raw[:512], hdrs=None, fp=None)

        return json.loads(raw)


_HTTP_CLIENT = _PersistentHTTPJSONClient()


def _retry_llm_call(func, retries: int = 3, delay: int = 2):
    """Retry wrapper with exponential backoff for transient LLM failures."""
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))

def _load_env_value_from_dotenv(key: str, dotenv_path: str = ".env") -> str:
    """Best-effort local .env loader for a single environment key."""
    if not os.path.exists(dotenv_path):
        return ""
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""



def _resolve_llm_config(default_model: str) -> tuple[str, str, str]:
    """Resolve API key (optional), base URL and model for local LLM servers."""
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip() or _load_env_value_from_dotenv("LOCAL_LLM_BASE_URL")
    if not base_url:
        base_url = DEFAULT_LOCAL_LLM_BASE_URL

    model = os.getenv("LOCAL_LLM_MODEL", "").strip() or _load_env_value_from_dotenv("LOCAL_LLM_MODEL")
    if not model:
        model = default_model

    api_key = os.getenv("LOCAL_LLM_API_KEY", "").strip() or _load_env_value_from_dotenv("LOCAL_LLM_API_KEY")
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
        "DOCUMENT TYPE": "DOCUMENT_TYPE",
        "DOCUMENTTYPE": "DOCUMENT_TYPE",
    }
    return aliases.get(norm, norm)

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


def _create_ner_completion(base_url: str, resolved_model: str, prompt: str, api_key: str = "") -> str:
    """Create completion against local LLM servers, trying common Ollama-compatible endpoints."""
    base = base_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
    }
    attempts = [
            (
                base + "/api/generate",
                {
                    "model": resolved_model,
                    "stream": False,
                    "format": "json",
                    "prompt": (
                        "You are a strict JSON information extraction engine. "
                        "Return JSON only.\n\n" + prompt
                    ),
                    "options": {"temperature": 0, "top_p": 1, "num_predict": 120},
                },
                lambda parsed: parsed.get("response", "{}"),
            ),
            (
                base + "/api/chat",
                {
                    "model": resolved_model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": "You are a strict JSON information extraction engine."},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0, "top_p": 1, "num_predict": 120},
                },
                lambda parsed: (
                    parsed.get("message", {}).get("content")
                    or parsed.get("response")
                    or parsed.get("content")
                    or "{}"
                ),
            ),
            (
                base + "/v1/chat/completions",
                {
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": "You are a strict JSON information extraction engine."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "top_p": 1,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 256,
                },
                lambda parsed: parsed.get("choices", [{}])[0].get("message", {}).get("content", "{}"),
            ),
        ]
        
    last_http_error = None
    for endpoint, payload, extract_content in attempts:
            try:
                parsed = _HTTP_CLIENT.post_json(endpoint, payload, headers=headers, timeout=120)
                content = extract_content(parsed)
                if isinstance(content, str) and content.strip():
                    return content
                return "{}"
            except error.HTTPError as e:
                if e.code == 404:
                    last_http_error = e
                    continue
                raise

    if last_http_error is not None:
            raise LLMNERConfigError(
                "Local LLM endpoint returned HTTP 404 Not Found on all supported paths. "
                f"Configured LOCAL_LLM_BASE_URL={base_url}. Check Ollama version and APIs: "
                "/api/chat, /api/generate, or /v1/chat/completions."
            ) from last_http_error

    raise LLMNERConfigError(
        "No local LLM completion endpoint succeeded. Verify LOCAL_LLM_BASE_URL and server logs."
    )


def extract_passport_fields_llm(ocr_json_path: str, model: str = DEFAULT_LOCAL_LLM_MODEL, regex_entities: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Dict[str, Any]]:
    """Extract dynamic passport/ID entities from OCR JSON using a local LLM server (default: Ollama)."""
    api_key, base_url, resolved_model = _resolve_llm_config(model)

    ocr_rows = load_ocr_json(ocr_json_path)
    filtered_rows = _filter_ocr_rows_for_llm(ocr_rows)
    prompt = build_passport_ner_prompt(filtered_rows or ocr_rows, regex_entities=regex_entities or {})
    
    try:
        content = _retry_llm_call(
            lambda: _create_ner_completion(base_url, resolved_model, prompt, api_key=api_key),
            retries=3,
            delay=2,
        )
    except LLMNERConfigError:
        raise
    except error.URLError as e:
        raise LLMNERConfigError(
            "Cannot reach local LLM server. Start Ollama and verify LOCAL_LLM_BASE_URL "
            f"(current: {base_url})."
        ) from e
    except Exception as e:
        msg = str(e)
        msg_lower = msg.lower()

        if "404" in msg_lower or "not found" in msg_lower:
            raise LLMNERConfigError(
                "Local LLM endpoint returned HTTP 404 Not Found. "
                f"Configured LOCAL_LLM_BASE_URL={base_url}. This usually means the server is reachable "
                "but does not expose expected Ollama APIs. Verify Ollama version and that /api/chat, "
                "/api/generate, or /v1/chat/completions is available. Also ensure LOCAL_LLM_BASE_URL "
                "does not already include /api or /v1."
            ) from e

        if "insufficient_quota" in msg or "error code: 429" in msg_lower or "quota" in msg_lower:
            raise LLMNERQuotaError(
                "LLM provider quota exceeded (429 insufficient_quota). "

                "Check provider limits, or continue with regex fallback."
            ) from e

        auth_markers = (
            "error code: 401",
            "unauthorized",
            "invalid api key",
            "incorrect api key",
            "forbidden",
        )
        if any(marker in msg_lower for marker in auth_markers):
            raise LLMNERAuthError(
                "LLM provider authentication failed (401/unauthorized). "
                "Verify LOCAL_LLM_API_KEY and endpoint access."
            ) from e

        token_limit_markers = (
            "maximum context length",
            "context length exceeded",
            "too many tokens",
            "token limit",
        )

        if any(marker in msg_lower for marker in token_limit_markers):
            raise LLMNERTokenLimitError(
                "LLM request exceeded model token/context limits. "
                "Reduce OCR payload size or use regex fallback."
            ) from e
        raise
 
    payload = _safe_parse_json(content)

    entities_raw = payload.get("entities", payload)
    entities_by_field: Dict[str, Dict[str, Any]] = {}

    # Accept both list-style payloads and dict-style payloads from different prompts/models.
    if isinstance(entities_raw, list):
        for item in entities_raw:
            if not isinstance(item, dict):
                continue
            field = _canonicalize_field_name(item.get("field", ""))
            if not field:
                continue
            entities_by_field[field] = {
                "text": item.get("text", ""),
                "bbox": item.get("bbox"),
                "confidence": item.get("confidence", 0.0),
            }
    elif isinstance(entities_raw, dict):
        for raw_field, item in entities_raw.items():
            field = _canonicalize_field_name(raw_field)
            if not field:
                continue
            if isinstance(item, dict):
                entities_by_field[field] = {
                    "text": item.get("text", ""),
                    "bbox": item.get("bbox"),
                    "confidence": item.get("confidence", 0.0),
                }
            else:
                entities_by_field[field] = {
                    "text": str(item),
                    "bbox": None,
                    "confidence": 0.0,
                }

    normalized_entities: Dict[str, Dict[str, Any]] = {}
    for field, item in entities_by_field.items():
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
