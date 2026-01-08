"""Data loaders for Phase 2: Terminal/Shell Mastery.

Datasets:
- Mitchins/NL-SHELL-MULTI (78K)
- westenfelder/NL2SH-ALFA (41K)
- b-mc2/cli-commands-explained (16K)
- neulab/tldr (9K)
"""

from datasets import load_dataset
from typing import Iterator, Dict, Any, Optional
import random


def load_nl_shell_multi(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load NL-SHELL-MULTI dataset.

    Combined dataset from NL2Bash + TLDR Pages + NL2SH-ALFA.
    Format: description/command pairs.
    """
    ds = load_dataset(
        "Mitchins/NL-SHELL-MULTI",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        try:
            desc = example.get("description") or example.get("nl") or example.get("query", "")
            cmd = example.get("command") or example.get("bash") or example.get("answer", "")

            if not desc or not cmd:
                continue

            instruction = f"Write a bash command to: {desc}"

            yield {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": cmd},
                ],
                "source": "nl_shell_multi",
            }
            count += 1
        except Exception:
            continue


def load_nl2sh_alfa(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load NL2SH-ALFA dataset.

    MIT-CSAIL dataset with verified test pairs including difficulty ratings.
    """
    ds = load_dataset(
        "westenfelder/NL2SH-ALFA",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        try:
            desc = example.get("nl") or example.get("description") or example.get("query", "")
            cmd = example.get("bash") or example.get("command") or example.get("answer", "")

            if not desc or not cmd:
                continue

            instruction = f"Write a bash command to: {desc}"

            yield {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": cmd},
                ],
                "source": "nl2sh_alfa",
                "difficulty": example.get("difficulty", "unknown"),
            }
            count += 1
        except Exception:
            continue


def load_cli_commands_explained(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load cli-commands-explained dataset.

    Commands from Commandlinefu with explanations.
    """
    ds = load_dataset(
        "b-mc2/cli-commands-explained",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Can use both directions: explanation->command or command->explanation
        # Using explanation->command for consistency

        instruction = f"Write a command that does the following:\n{example['explanation']}"

        yield {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": example["command"]},
            ],
            "source": "cli_commands_explained",
        }
        count += 1


def load_tldr(streaming: bool = False, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Load TLDR pages dataset.

    NL-to-bash with manual references from TLDR pages project.
    """
    ds = load_dataset(
        "neulab/tldr",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        yield {
            "messages": [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["answer"]},
            ],
            "source": "tldr",
        }
        count += 1


def load_terminal_datasets(
    streaming: bool = False,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """Load all Phase 2 terminal datasets.

    Args:
        streaming: Whether to stream datasets
        max_samples_per_dataset: Max samples from each dataset
        shuffle: Whether to shuffle datasets
        seed: Random seed for shuffling

    Yields:
        Dict with 'messages' and 'source'
    """
    # Load all datasets
    nl_shell = list(load_nl_shell_multi(streaming=streaming, max_samples=max_samples_per_dataset))
    nl2sh = list(load_nl2sh_alfa(streaming=streaming, max_samples=max_samples_per_dataset))
    cli_cmds = list(load_cli_commands_explained(streaming=streaming, max_samples=max_samples_per_dataset))
    tldr_data = list(load_tldr(streaming=streaming, max_samples=max_samples_per_dataset))

    all_data = nl_shell + nl2sh + cli_cmds + tldr_data

    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)

    for item in all_data:
        yield item


def get_terminal_dataset_info() -> Dict[str, Any]:
    """Get information about terminal datasets."""
    return {
        "phase": 2,
        "name": "Terminal/Shell Mastery",
        "datasets": [
            {
                "name": "NL-SHELL-MULTI",
                "hf_path": "Mitchins/NL-SHELL-MULTI",
                "size": 78768,
                "format": "description/command",
            },
            {
                "name": "NL2SH-ALFA",
                "hf_path": "westenfelder/NL2SH-ALFA",
                "size": 40939,
                "format": "nl/bash with difficulty",
            },
            {
                "name": "cli-commands-explained",
                "hf_path": "b-mc2/cli-commands-explained",
                "size": 16098,
                "format": "command/explanation",
            },
            {
                "name": "TLDR",
                "hf_path": "neulab/tldr",
                "size": 9187,
                "format": "query/answer",
            },
        ],
        "total_estimated": "~145K examples",
    }
