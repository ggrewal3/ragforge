"""Small Gemini adapter: one generate() call with timeouts, bounded retries and clear errors.

The rest of the project only depends on the interface

    llm.model                                   model ID string
    llm.generate(system_instruction, user_prompt, response_json_schema) -> LLMResponse

so tests can pass a fake object with the same shape, and retrieval never
imports the Gemini SDK.

Configuration (environment variables, or a local .env file loaded by the CLI):
    GEMINI_API_KEY   required for live calls; never printed or logged
    GEMINI_MODEL     optional model ID, default DEFAULT_GEMINI_MODEL

API choice: this uses `client.models.generate_content` from the official
google-genai SDK. Google recommends its newer Interactions API for new
projects but labels it beta and subject to breaking changes, while
generateContent "remains fully supported" and is the recommendation for
stable use.
"""

import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

API_KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "GEMINI_MODEL"

# Current stable Flash model per https://ai.google.dev/gemini-api/docs/models (checked 2026-10-19).
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 2          # retries after the first attempt, for transient failures only
RETRY_BASE_DELAY_S = 1.0         # 1 s, then 2 s
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Paid-tier text prices in USD per 1M tokens, from https://ai.google.dev/gemini-api/docs/pricing
# (retrieved 2026-10-19). Output prices include thinking tokens. Each entry is only trusted
# until `valid_until`; after that no estimate is produced. The free tier costs nothing.
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing (retrieved 2026-10-19)"
PRICING = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "valid_until": date(2026, 12, 31)},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75, "valid_until": date(2026, 12, 31)},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "valid_until": None},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "valid_until": None},
}

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for generation failures. `kind` is a short machine-readable label."""

    kind = "llm_error"


class LLMConfigurationError(LLMError):
    kind = "configuration"


class LLMAuthenticationError(LLMError):
    kind = "authentication"


class LLMRateLimitError(LLMError):
    kind = "rate_limit"


class LLMTimeoutError(LLMError):
    kind = "timeout"


class LLMRequestError(LLMError):
    kind = "api_error"


class LLMEmptyResponseError(LLMError):
    kind = "empty_response"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_env_file(path):
    """Load KEY=VALUE lines from a .env file into os.environ without overriding existing variables.

    Supports blank lines, '#' comments, an optional 'export ' prefix and single
    or double quotes around values. Returns the names that were set (never values).
    """
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def resolve_model(model=None):
    """Model ID from the argument, then GEMINI_MODEL, then the default; validated for shape only."""
    model = model or os.environ.get(MODEL_ENV) or DEFAULT_GEMINI_MODEL
    model = model.strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    if not _MODEL_ID_PATTERN.match(model):
        raise LLMConfigurationError(f"Invalid Gemini model ID {model!r}; expected something like '{DEFAULT_GEMINI_MODEL}'.")
    return model


def validate_sampling(temperature=None, top_p=None):
    """Range checks only. None means "do not send; use the model's default"."""
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
            raise LLMConfigurationError(f"temperature must be a number in [0, 2], got {temperature!r}.")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0.0 < top_p <= 1.0:
            raise LLMConfigurationError(f"top_p must be a number in (0, 1], got {top_p!r}.")


def sampling_warnings(model, temperature=None, top_p=None):
    """Documented caveats for sampling parameters on Gemini 3-series models (advice, not enforcement)."""
    warnings = []
    if model.startswith("gemini-3") and (temperature is not None or top_p is not None):
        warnings.append(
            "Google's Gemini 3 guidance recommends leaving temperature at its default (1.0); values below 1.0 "
            "'may lead to unexpected behavior, such as looping or degraded performance'. The Gemini 3.8 Flash "
            "migration guide lists temperature/top_p/top_k as deprecated sampling parameters.")
    return warnings


def estimate_cost_usd(model, input_tokens, output_tokens, today=None):
    """Paid-tier cost estimate, or None when no trusted price is known for the model or date."""
    price = PRICING.get(model)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    today = today or date.today()
    if price["valid_until"] is not None and today > price["valid_until"]:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    attempts: int
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)  # input_tokens, output_tokens, thinking_tokens, total_tokens


def _usage(response):
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {}
    candidates = getattr(metadata, "candidates_token_count", None)
    thoughts = getattr(metadata, "thoughts_token_count", None)
    billed_output = None if candidates is None and thoughts is None else (candidates or 0) + (thoughts or 0)
    return {
        "input_tokens": getattr(metadata, "prompt_token_count", None),
        "output_tokens": billed_output,          # answer + thinking, both billed at the output rate
        "thinking_tokens": thoughts,
        "total_tokens": getattr(metadata, "total_token_count", None),
    }


class GeminiClient:
    """Thin wrapper around google-genai's generate_content with bounded retries."""

    def __init__(self, model=None, api_key=None, temperature=None, top_p=None, timeout_s=DEFAULT_TIMEOUT_S,
                 max_retries=DEFAULT_MAX_RETRIES, sdk_client=None, sleep=time.sleep):
        self.model = resolve_model(model)
        validate_sampling(temperature, top_p)
        if not (isinstance(timeout_s, (int, float)) and timeout_s > 0):
            raise LLMConfigurationError(f"timeout_s must be positive, got {timeout_s!r}.")
        if not (isinstance(max_retries, int) and 0 <= max_retries <= 5):
            raise LLMConfigurationError(f"max_retries must be an integer in [0, 5], got {max_retries!r}.")
        self.temperature, self.top_p = temperature, top_p
        self.timeout_s, self.max_retries = float(timeout_s), max_retries
        self._sleep = sleep
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if sdk_client is not None:
            self._client = sdk_client
        else:
            if not self._api_key:
                raise LLMConfigurationError(
                    f"{API_KEY_ENV} is not set. Export it or put it in a local .env file (see .env.example).")
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout_s * 1000),               # milliseconds
                    retry_options=types.HttpRetryOptions(attempts=1),  # retries are handled here, not in the SDK
                ),
            )

    def __repr__(self):  # never include the key
        return f"GeminiClient(model={self.model!r}, temperature={self.temperature!r}, top_p={self.top_p!r})"

    def _redact(self, message):
        message = str(message)
        if self._api_key:
            message = message.replace(self._api_key, "***")
        return message

    def _config(self, system_instruction, response_json_schema):
        from google.genai import types

        options = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_json_schema": response_json_schema,
        }
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        return types.GenerateContentConfig(**options)

    def _classify(self, error):
        """Map an exception to (LLMError, retryable)."""
        from google.genai import errors

        if isinstance(error, errors.APIError):
            code = getattr(error, "code", None)
            detail = self._redact(f"{code} {getattr(error, 'status', '')}: {getattr(error, 'message', '') or ''}")
            if code in (401, 403):
                return LLMAuthenticationError(f"Gemini rejected the credentials ({detail}). Check {API_KEY_ENV}."), False
            if code == 404:
                return LLMConfigurationError(f"Model {self.model!r} was not found ({detail}). Check {MODEL_ENV}."), False
            if code == 429:
                return LLMRateLimitError(f"Gemini rate limit or quota exceeded ({detail})."), True
            if code in RETRYABLE_STATUS_CODES:
                return LLMRequestError(f"Gemini server error ({detail})."), True
            if code == 400 and "API key" in detail:
                return LLMAuthenticationError(f"Gemini rejected the API key ({detail}). Check {API_KEY_ENV}."), False
            return LLMRequestError(f"Gemini request failed ({detail})."), False
        import httpx

        if isinstance(error, httpx.TimeoutException):
            return LLMTimeoutError(f"Gemini request timed out after {self.timeout_s:g} s."), True
        if isinstance(error, httpx.TransportError):
            return LLMRequestError(f"Network error calling Gemini: {self._redact(type(error).__name__)}."), True
        return LLMRequestError(f"Unexpected error calling Gemini: {self._redact(f'{type(error).__name__}: {error}')}"), False

    def generate(self, system_instruction, user_prompt, response_json_schema):
        """One structured-output request. Raises an LLMError subclass on failure."""
        config = self._config(system_instruction, response_json_schema)
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._client.models.generate_content(model=self.model, contents=user_prompt, config=config)
                break
            except Exception as error:  # noqa: BLE001 - classified and re-raised below
                llm_error, retryable = self._classify(error)
                if not retryable or attempt > self.max_retries:
                    raise llm_error from None
                self._sleep(RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))

        latency = time.perf_counter() - start
        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        if candidates and getattr(candidates[0], "finish_reason", None) is not None:
            finish_reason = str(getattr(candidates[0].finish_reason, "name", candidates[0].finish_reason))
        text = getattr(response, "text", None)
        if not text:
            feedback = getattr(response, "prompt_feedback", None)
            reason = finish_reason or (getattr(feedback, "block_reason", None) if feedback else None) or "unknown"
            raise LLMEmptyResponseError(f"Gemini returned no text (finish reason: {reason}).")
        return LLMResponse(text=text, model=self.model, latency_s=latency, attempts=attempt,
                           finish_reason=finish_reason, usage=_usage(response))
