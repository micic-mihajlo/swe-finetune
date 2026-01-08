"""Data loaders for Phase 4: SWE-bench Agent Trajectories.

Datasets:
- nebius/SWE-agent-trajectories (80K)
- SWE-bench/SWE-smith-trajectories (76K)
- nebius/SWE-rebench-openhands-trajectories (for OpenHands format)

This is the CRITICAL phase for SWE-bench performance.
"""

from datasets import load_dataset
from typing import Iterator, Dict, Any, Optional, List
import json
import random


def parse_trajectory_to_messages(trajectory: Any) -> List[Dict[str, str]]:
    """Parse agent trajectory into conversation messages.

    Trajectories typically have step-by-step format:
    - observation: what the agent sees (user turn)
    - action: what the agent does (assistant turn)
    """
    messages = []

    # Handle different trajectory formats
    if isinstance(trajectory, str):
        try:
            trajectory = json.loads(trajectory)
        except json.JSONDecodeError:
            # Plain text trajectory
            return [{"role": "assistant", "content": trajectory}]

    if isinstance(trajectory, list):
        for step in trajectory:
            if isinstance(step, dict):
                # Standard format: observation/action pairs
                if "observation" in step:
                    messages.append({"role": "user", "content": step["observation"]})
                if "action" in step:
                    messages.append({"role": "assistant", "content": step["action"]})
                # Alternative format: role/content
                elif "role" in step and "content" in step:
                    messages.append(step)
            elif isinstance(step, str):
                # Alternating strings (assume user/assistant)
                role = "user" if len(messages) % 2 == 0 else "assistant"
                messages.append({"role": role, "content": step})

    elif isinstance(trajectory, dict):
        # Single step or nested format
        if "messages" in trajectory:
            messages = trajectory["messages"]
        elif "steps" in trajectory:
            return parse_trajectory_to_messages(trajectory["steps"])
        elif "observation" in trajectory and "action" in trajectory:
            messages.append({"role": "user", "content": trajectory["observation"]})
            messages.append({"role": "assistant", "content": trajectory["action"]})

    return messages


def load_swe_agent_trajectories(
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Load nebius/SWE-agent-trajectories dataset.

    80K trajectories from SWE-agent framework.
    """
    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Parse trajectory
        trajectory = example.get("trajectory", example.get("messages", []))
        messages = parse_trajectory_to_messages(trajectory)

        if not messages:
            continue

        # Add problem context if available
        problem = example.get("problem_statement", example.get("issue", ""))
        if problem and messages[0]["role"] != "system":
            messages.insert(0, {
                "role": "system",
                "content": f"You are a software engineer. Solve the following issue:\n\n{problem}"
            })

        yield {
            "messages": messages,
            "source": "swe_agent_trajectories",
            "instance_id": example.get("instance_id", ""),
            "resolved": example.get("resolved", None),
        }
        count += 1


def load_swe_smith_trajectories(
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Load SWE-bench/SWE-smith-trajectories dataset.

    76K Claude 3.7 Sonnet traces.
    """
    ds = load_dataset(
        "SWE-bench/SWE-smith-trajectories",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Parse trajectory
        trajectory = example.get("trajectory", example.get("messages", []))
        messages = parse_trajectory_to_messages(trajectory)

        if not messages:
            continue

        # Add problem context if available
        problem = example.get("problem_statement", example.get("issue", ""))
        if problem and messages[0]["role"] != "system":
            messages.insert(0, {
                "role": "system",
                "content": f"You are a software engineer. Solve the following issue:\n\n{problem}"
            })

        yield {
            "messages": messages,
            "source": "swe_smith_trajectories",
            "instance_id": example.get("instance_id", ""),
            "resolved": example.get("resolved", None),
        }
        count += 1


def load_openhands_trajectories(
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Load OpenHands format trajectories.

    Use this if running inference with OpenHands scaffolding.
    """
    try:
        ds = load_dataset(
            "nebius/SWE-rebench-openhands-trajectories",
            split="train",
            streaming=streaming,
        )
    except Exception as e:
        print(f"Warning: Could not load OpenHands trajectories: {e}")
        return

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        trajectory = example.get("trajectory", example.get("messages", []))
        messages = parse_trajectory_to_messages(trajectory)

        if not messages:
            continue

        yield {
            "messages": messages,
            "source": "openhands_trajectories",
            "instance_id": example.get("instance_id", ""),
        }
        count += 1


def load_swebench_trajectories(
    streaming: bool = True,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    format: str = "swe_agent",  # "swe_agent" or "openhands"
) -> Iterator[Dict[str, Any]]:
    """Load all Phase 4 SWE-bench trajectory datasets.

    Args:
        streaming: Whether to stream datasets
        max_samples_per_dataset: Max samples from each dataset
        shuffle: Whether to shuffle datasets
        seed: Random seed for shuffling
        format: "swe_agent" for SWE-agent format, "openhands" for OpenHands format

    Yields:
        Dict with 'messages', 'source', and metadata
    """
    if format == "openhands":
        # Only load OpenHands format
        all_data = list(load_openhands_trajectories(
            streaming=streaming,
            max_samples=max_samples_per_dataset,
        ))
    else:
        # Load SWE-agent format trajectories
        swe_agent = list(load_swe_agent_trajectories(
            streaming=streaming,
            max_samples=max_samples_per_dataset,
        ))
        swe_smith = list(load_swe_smith_trajectories(
            streaming=streaming,
            max_samples=max_samples_per_dataset,
        ))
        all_data = swe_agent + swe_smith

    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)

    for item in all_data:
        yield item


def get_swebench_dataset_info() -> Dict[str, Any]:
    """Get information about SWE-bench datasets."""
    return {
        "phase": 4,
        "name": "SWE-bench Agent Trajectories (CRITICAL)",
        "datasets": [
            {
                "name": "SWE-agent-trajectories",
                "hf_path": "nebius/SWE-agent-trajectories",
                "size": 80036,
                "format": "step-by-step agent actions",
                "benchmark": "40.6% on SWE-bench Verified",
            },
            {
                "name": "SWE-smith-trajectories",
                "hf_path": "SWE-bench/SWE-smith-trajectories",
                "size": 76000,
                "format": "Claude 3.7 Sonnet traces",
                "benchmark": "40.2% on SWE-bench Verified",
            },
            {
                "name": "OpenHands-trajectories (optional)",
                "hf_path": "nebius/SWE-rebench-openhands-trajectories",
                "size": "varies",
                "format": "OpenHands agent format",
                "note": "Use if running inference with OpenHands",
            },
        ],
        "total_estimated": "~156K examples",
        "critical": True,
    }
