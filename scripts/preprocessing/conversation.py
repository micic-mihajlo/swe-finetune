"""Conversation format utilities."""

from typing import Dict, Any, List


def convert_to_messages(example: Dict[str, Any], format_type: str = "instruction") -> List[Dict[str, str]]:
    """Convert dataset example to standardized message format."""
    if format_type == "instruction":
        inst = example.get("instruction", example.get("query", example.get("question", "")))
        resp = example.get("response", example.get("output", example.get("answer", "")))
        if inst and resp:
            return [{"role": "user", "content": inst}, {"role": "assistant", "content": resp}]
    elif format_type == "conversation":
        msgs = example.get("messages", example.get("conversations", []))
        return [{"role": m.get("role", m.get("from", "")).replace("human", "user").replace("gpt", "assistant"),
                 "content": m.get("content", m.get("value", ""))} for m in msgs if m]
    return []


def validate_messages(messages: List[Dict[str, str]]) -> bool:
    """Validate message format."""
    if not messages:
        return False
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in {"user", "assistant", "system"}:
            return False
    return True


def truncate_messages(messages: List[Dict[str, str]], max_chars: int = 100000) -> List[Dict[str, str]]:
    """Truncate messages to fit within character limit."""
    total = sum(len(m["content"]) for m in messages)
    if total <= max_chars:
        return messages
    # Keep system + last 2 messages minimum
    result = []
    if messages and messages[0]["role"] == "system":
        result.append(messages[0])
        messages = messages[1:]
    return result + messages[-2:]
