"""Data loaders for Phase 1: General Coding Foundation.

Datasets:
- ise-uiuc/Magicoder-OSS-Instruct-75K
- nickrosh/Evol-Instruct-Code-80k-v1
- bigcode/commitpackft
"""

from datasets import load_dataset, interleave_datasets
from typing import Iterator, Dict, Any, Optional
import random


def load_magicoder(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load Magicoder-OSS-Instruct-75K dataset.

    Format: instruction/response pairs for coding tasks.
    """
    ds = load_dataset(
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        yield {
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]},
            ],
            "source": "magicoder",
        }
        count += 1


def load_evol_instruct(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load Evol-Instruct-Code-80k dataset.

    Format: instruction/output pairs evolved from Code Alpaca.
    """
    ds = load_dataset(
        "nickrosh/Evol-Instruct-Code-80k-v1",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        yield {
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ],
            "source": "evol_instruct",
        }
        count += 1


def load_commitpackft(
    streaming: bool = True,
    max_samples: Optional[int] = 50000,
    lang: str = "Python",
) -> Iterator[Dict[str, Any]]:
    """Load CommitPackFT dataset (Python subset).

    Format: commit message as instruction, code diff as response.
    """
    ds = load_dataset(
        "bigcode/commitpackft",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Filter by language if specified
        if lang and example.get("lang", "") != lang:
            continue

        # Format commit as instruction
        instruction = f"Apply the following change to the codebase:\n\n{example['message']}"
        response = example.get("content", example.get("patch", ""))

        if not response:
            continue

        yield {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "source": "commitpackft",
        }
        count += 1


def load_coding_datasets(
    streaming: bool = False,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """Load all Phase 1 coding datasets.

    Args:
        streaming: Whether to stream datasets
        max_samples_per_dataset: Max samples from each dataset
        shuffle: Whether to shuffle (interleave) datasets
        seed: Random seed for shuffling

    Yields:
        Dict with 'messages' (list of role/content dicts) and 'source' (dataset name)
    """
    # Load all datasets
    magicoder = list(load_magicoder(streaming=streaming, max_samples=max_samples_per_dataset))
    evol = list(load_evol_instruct(streaming=streaming, max_samples=max_samples_per_dataset))
    commits = list(load_commitpackft(streaming=True, max_samples=max_samples_per_dataset or 50000))

    all_data = magicoder + evol + commits

    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)

    for item in all_data:
        yield item


def get_coding_dataset_info() -> Dict[str, Any]:
    """Get information about coding datasets."""
    return {
        "phase": 1,
        "name": "General Coding Foundation",
        "datasets": [
            {
                "name": "Magicoder-OSS-Instruct-75K",
                "hf_path": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                "size": 75000,
                "format": "instruction/response",
            },
            {
                "name": "Evol-Instruct-Code-80k",
                "hf_path": "nickrosh/Evol-Instruct-Code-80k-v1",
                "size": 80000,
                "format": "instruction/output",
            },
            {
                "name": "CommitPackFT",
                "hf_path": "bigcode/commitpackft",
                "size": "~50K (Python subset)",
                "format": "commit_message/diff",
            },
        ],
        "total_estimated": "~205K examples",
    }
