"""Offline unit tests for the Bedrock LLM client — no boto3 client, no network.

Only the pure request/response plumbing is exercised here (JSON extraction, backend
selection, message shaping, the complete_json retry loop). The live smoke test against
Bedrock lives in run_llm_example, not the deterministic suite.
"""
from __future__ import annotations

import pytest

from ajar.llm.bedrock import (
    BedrockLLM,
    _balanced_span,
    _extract_json,
    _flatten_prompt,
    _mantle_text,
    _resolve_backend,
    _split_system,
)


def test_extract_json_plain():
    assert _extract_json('[1, 2, 3]') == [1, 2, 3]
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_prose():
    assert _extract_json("```json\n[{\"v\": 1}]\n```") == [{"v": 1}]
    assert _extract_json('Sure, here you go: {"a": [1, 2]} — done!') == {"a": [1, 2]}


def test_extract_json_ignores_braces_inside_strings():
    # A closing brace inside a string must not end the span early.
    assert _extract_json('{"msg": "a } b", "n": 1}') == {"msg": "a } b", "n": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")
    with pytest.raises(ValueError):
        _extract_json("")


def test_balanced_span_unterminated():
    assert _balanced_span('{"a": 1', 0) is None


def test_resolve_backend():
    assert _resolve_backend("auto", "us.anthropic.claude-sonnet-4-20250514-v1:0") == "converse"
    assert _resolve_backend("auto", "openai.gpt-5.5") == "responses"
    assert _resolve_backend("converse", "openai.gpt-5.5") == "converse"  # explicit wins
    with pytest.raises(ValueError):
        _resolve_backend("nonsense", "x")


def test_split_system_separates_and_never_empties():
    system, turns = _split_system(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert system == "S"
    assert turns == [{"role": "user", "content": [{"text": "U"}]}]
    # Converse rejects an empty message list, so a system-only convo still yields a turn.
    _, only = _split_system([{"role": "system", "content": "S"}])
    assert only and only[0]["role"] == "user"


def test_flatten_prompt_role_labelled():
    assert _flatten_prompt([{"role": "user", "content": "hi"}]) == "USER: hi"


def test_mantle_text_joins_output_text_only():
    payload = {
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "think"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "ans"}, {"type": "output_text", "text": "wer"}]},
        ]
    }
    assert _mantle_text(payload) == "answer"


class _ScriptedLLM(BedrockLLM):
    """BedrockLLM with `complete` stubbed to a fixed reply sequence — exercises the
    complete_json parse/retry loop without touching Bedrock."""

    def __init__(self, replies):
        super().__init__()
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        return self._replies.pop(0)


def test_complete_json_retries_then_parses():
    llm = _ScriptedLLM(["not json", '```json\n[1, 2]\n```'])
    assert llm.complete_json([{"role": "user", "content": "x"}], {"type": "array"}) == [1, 2]
    assert llm.calls == 2  # one failure, one success


def test_complete_json_gives_up_with_typed_empty():
    # 3 replies = the initial call + 2 retries, all unparseable.
    assert _ScriptedLLM(["a", "b", "c"]).complete_json([{"role": "user", "content": "x"}], {"type": "array"}) == []
    assert _ScriptedLLM(["a", "b", "c"]).complete_json([{"role": "user", "content": "x"}], {"type": "object"}) == {}
