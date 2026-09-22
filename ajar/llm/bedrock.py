"""A concrete `LLMClient` backed by AWS Bedrock — no API key, IAM-role auth.

This is an INJECTED dependency (see pyproject `bedrock` extra): `boto3` is imported
lazily inside the methods, never at module scope, so `import ajar` still works in
a bare interpreter and the core stays stdlib-only.

Two backends, auto-selected from the model id:

* **converse** (default) — `bedrock-runtime.converse`, used for Anthropic Claude and
  any other Converse-API model. Default model: Claude Sonnet 4.
* **responses** — the `bedrock-mantle` OpenAI-compatible `/responses` endpoint,
  SigV4-signed with the instance role, used for `openai.*` models (e.g. GPT-5.5).

Both implement the `LLMClient` protocol (`complete` / `complete_json`). Per the seam's
design rule the client only *proposes*: whatever it returns is disposed by the oracle
before it becomes a probe label, so a flaky completion can never mislabel a probe — it
can only widen or narrow coverage.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

# Confirmed live on this box (bedrock us-east-2): converse returns "PONG" cleanly.
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_REGION = "us-east-2"
_MANTLE_URL = "https://bedrock-mantle.{region}.api.aws/openai/v1/responses"


class BedrockLLM:
    """`LLMClient` over Bedrock. Pick a backend explicitly or let the model id decide.

        BedrockLLM()                         # Claude Sonnet 4, converse
        BedrockLLM(model="openai.gpt-5.5")   # GPT-5.5, mantle /responses
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        region: str = DEFAULT_REGION,
        backend: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_json_retries: int = 2,
    ) -> None:
        self.model = model
        self.region = region
        self.backend = _resolve_backend(backend, model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_json_retries = max_json_retries
        self._runtime = None  # lazy boto3 client (converse backend)

    # -- LLMClient protocol --------------------------------------------------

    def complete(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        if self.backend == "converse":
            return self._converse(messages, **kwargs)
        return self._responses(messages, **kwargs)

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Ask for JSON conforming to `schema`, parse it, retry on malformed output.

        Neither backend enforces a JSON schema server-side, so we instruct-and-parse:
        append the schema to the prompt, extract the first JSON value from the reply,
        and re-ask with the parser error on failure. The top-level `type` in `schema`
        picks the empty fallback so a give-up still satisfies "no proposals".
        """
        want_array = (schema or {}).get("type") == "array"
        convo = list(messages) + [{"role": "user", "content": _json_instruction(schema)}]
        last_err = ""
        for attempt in range(self.max_json_retries + 1):
            text = self.complete(convo, **kwargs)
            try:
                return _extract_json(text)
            except ValueError as e:
                last_err = str(e)
                convo = convo + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That was not valid JSON ({last_err}). Reply with ONLY the JSON value."},
                ]
        # Exhausted retries: an empty container satisfies "no proposals".
        return [] if want_array else {}

    # -- backends ------------------------------------------------------------

    def _converse(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        import boto3  # lazy: keeps `import ajar` stdlib-only

        if self._runtime is None:
            self._runtime = boto3.client("bedrock-runtime", region_name=self.region)

        system, turns = _split_system(messages)
        req: dict[str, Any] = {
            "modelId": self.model,
            "messages": turns,
            "inferenceConfig": {
                "maxTokens": int(kwargs.get("max_tokens", self.max_tokens)),
                "temperature": float(kwargs.get("temperature", self.temperature)),
            },
        }
        if system:
            req["system"] = [{"text": system}]
        resp = self._runtime.converse(**req)
        blocks = resp["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks)

    def _responses(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        import urllib.request

        import boto3  # lazy
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        creds = boto3.Session().get_credentials().get_frozen_credentials()
        url = _MANTLE_URL.format(region=self.region)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": _flatten_prompt(messages),
            "max_output_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
        }
        body = json.dumps(payload)
        signed = AWSRequest("POST", url, data=body, headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock", self.region).add_auth(signed)
        req = urllib.request.Request(url, data=body.encode(), headers=dict(signed.headers), method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return _mantle_text(data)


# -- module helpers ---------------------------------------------------------


def _resolve_backend(backend: str, model: str) -> str:
    if backend != "auto":
        if backend not in ("converse", "responses"):
            raise ValueError(f"unknown backend {backend!r} (converse | responses | auto)")
        return backend
    return "responses" if model.startswith("openai.") else "converse"


def _split_system(messages: Sequence[Mapping[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    """Converse takes `system` separately and only user/assistant turns in `messages`."""
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            turns.append({"role": role, "content": [{"text": content}]})
    if not turns:  # Converse rejects an empty message list.
        turns = [{"role": "user", "content": [{"text": ""}]}]
    return "\n\n".join(system_parts), turns


def _flatten_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """The /responses `input` is sent as one role-labelled string (smoke-tested)."""
    return "\n\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages)


def _mantle_text(data: Mapping[str, Any]) -> str:
    """Join every `output_text` block in a mantle /responses payload."""
    parts: list[str] = []
    for block in data.get("output", []) or []:
        for item in block.get("content", []) or []:
            if item.get("type") == "output_text":
                parts.append(item.get("text", ""))
    return "".join(parts)


def _json_instruction(schema: Mapping[str, Any]) -> str:
    return (
        "Return ONLY a single JSON value (no prose, no markdown fences) that conforms "
        "to this JSON schema:\n" + json.dumps(schema, indent=2)
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """First JSON value in `text`, tolerating markdown fences and surrounding prose."""
    if not text or not text.strip():
        raise ValueError("empty completion")
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate.strip())
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} or [...] span.
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON object/array found in completion")
    snippet = _balanced_span(candidate, start)
    if snippet is None:
        raise ValueError("unterminated JSON value in completion")
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e}") from e


def _balanced_span(s: str, start: int) -> str | None:
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None
