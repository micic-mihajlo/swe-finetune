"""Data loaders for Phase 3: Tool-Use & Function Calling.

Datasets:
- Salesforce/xlam-function-calling-60k
- glaiveai/glaive-function-calling-v2
- NousResearch/hermes-function-calling-v1
"""

from datasets import load_dataset
from typing import Iterator, Dict, Any, Optional, List
import json
import random


def parse_glaive_format(text: str) -> List[Dict[str, str]]:
    """Parse Glaive function calling format into messages.

    Format: SYSTEM: ... USER: ... ASSISTANT: ... <functioncall> ... FUNCTION RESPONSE: ...
    """
    messages = []

    # Split by role markers
    parts = text.split("SYSTEM:")
    if len(parts) > 1:
        system_rest = parts[1].split("USER:")
        if system_rest[0].strip():
            messages.append({"role": "system", "content": system_rest[0].strip()})

    # Find USER/ASSISTANT pairs
    if "USER:" in text:
        user_parts = text.split("USER:")[1:]
        for user_part in user_parts:
            if "ASSISTANT:" in user_part:
                user_content, assistant_part = user_part.split("ASSISTANT:", 1)
                messages.append({"role": "user", "content": user_content.strip()})

                # Handle function calls in assistant response
                assistant_content = assistant_part.split("FUNCTION RESPONSE:")[0].strip()
                messages.append({"role": "assistant", "content": assistant_content})

    return messages if messages else None


def load_xlam_function_calling(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load xLAM function calling dataset.

    Top-tier function calling dataset (Berkeley BFCL #3).
    """
    ds = load_dataset(
        "Salesforce/xlam-function-calling-60k",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # xLAM uses conversation format
        messages = example.get("conversations", [])
        if not messages:
            continue

        # Convert to standard format if needed
        formatted_messages = []
        for msg in messages:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))

            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"

            formatted_messages.append({"role": role, "content": content})

        if formatted_messages:
            yield {
                "messages": formatted_messages,
                "source": "xlam_function_calling",
            }
            count += 1


def load_glaive_function_calling(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load Glaive function calling v2 dataset.

    Large function calling dataset with custom format.
    """
    ds = load_dataset(
        "glaiveai/glaive-function-calling-v2",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Parse Glaive format
        text = example.get("chat", example.get("text", ""))
        messages = parse_glaive_format(text)

        if messages:
            yield {
                "messages": messages,
                "source": "glaive_function_calling",
            }
            count += 1


def load_hermes_function_calling(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load Hermes function calling dataset.

    ShareGPT format with <tool_call> and <tool_response> tags.
    """
    try:
        ds = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            split="train",
            streaming=streaming,
        )
    except Exception:
        # Dataset might have multiple files
        ds = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            data_files="*.json",
            split="train",
            streaming=streaming,
        )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # ShareGPT format
        conversations = example.get("conversations", [])
        if not conversations:
            continue

        messages = []
        for conv in conversations:
            role = conv.get("from", "")
            content = conv.get("value", "")

            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                continue

            messages.append({"role": role, "content": content})

        if messages:
            yield {
                "messages": messages,
                "source": "hermes_function_calling",
            }
            count += 1


def load_tooluse_datasets(
    streaming: bool = False,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """Load all Phase 3 tool-use datasets.

    Args:
        streaming: Whether to stream datasets
        max_samples_per_dataset: Max samples from each dataset
        shuffle: Whether to shuffle datasets
        seed: Random seed for shuffling

    Yields:
        Dict with 'messages' and 'source'
    """
    # Load all datasets
    xlam = list(load_xlam_function_calling(streaming=streaming, max_samples=max_samples_per_dataset))
    glaive = list(load_glaive_function_calling(streaming=streaming, max_samples=max_samples_per_dataset))
    hermes = list(load_hermes_function_calling(streaming=streaming, max_samples=max_samples_per_dataset))

    all_data = xlam + glaive + hermes

    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)

    for item in all_data:
        yield item


def get_tooluse_dataset_info() -> Dict[str, Any]:
    """Get information about tool-use datasets."""
    return {
        "phase": 3,
        "name": "Tool-Use & Function Calling",
        "datasets": [
            {
                "name": "xLAM-function-calling-60k",
                "hf_path": "Salesforce/xlam-function-calling-60k",
                "size": 60000,
                "format": "conversations with function calls",
            },
            {
                "name": "Glaive-function-calling-v2",
                "hf_path": "glaiveai/glaive-function-calling-v2",
                "size": 112960,
                "format": "SYSTEM/USER/ASSISTANT with <functioncall>",
            },
            {
                "name": "Hermes-function-calling-v1",
                "hf_path": "NousResearch/hermes-function-calling-v1",
                "size": "~5K+",
                "format": "ShareGPT with <tool_call> tags",
            },
        ],
        "total_estimated": "~180K examples",
    }
