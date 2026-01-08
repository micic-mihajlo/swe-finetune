"""Data loaders for Phase 5: Competitive Programming.

Datasets:
- BAAI/TACO (25K problems with algorithm labels)
- open-r1/codeforces-cots (10K problems + 100K reasoning traces)

Goal: Strengthen algorithmic reasoning and problem-solving.
"""

from datasets import load_dataset
from typing import Iterator, Dict, Any, Optional, List
import random


def load_taco(
    streaming: bool = False,
    max_samples: Optional[int] = None,
    difficulty: Optional[str] = None,  # EASY, MEDIUM, MEDIUM_HARD, HARD, VERY_HARD
) -> Iterator[Dict[str, Any]]:
    """Load BAAI/TACO dataset.

    25K problems with fine-grained algorithm labels.
    Includes solutions in multiple languages.
    """
    ds = load_dataset(
        "BAAI/TACO",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Filter by difficulty if specified
        if difficulty and example.get("difficulty", "") != difficulty:
            continue

        # Get problem and solution
        question = example.get("question", "")
        solutions = example.get("solutions", [])

        if not question or not solutions:
            continue

        # Use first solution (typically Python)
        solution = solutions[0] if isinstance(solutions, list) else solutions

        # Build instruction with problem context
        instruction = f"Solve the following competitive programming problem:\n\n{question}"

        # Add algorithm hints if available
        skills = example.get("skill_types", [])
        if skills:
            instruction += f"\n\nRelevant concepts: {', '.join(skills)}"

        yield {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": solution},
            ],
            "source": "taco",
            "difficulty": example.get("difficulty", ""),
            "skills": skills,
        }
        count += 1


def load_codeforces_cots(
    streaming: bool = False,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Load CodeForces Chain-of-Thought dataset.

    10K problems with ~100K reasoning traces from DeepSeek R1.
    """
    ds = load_dataset(
        "open-r1/codeforces-cots",
        split="train",
        streaming=streaming,
    )

    count = 0
    for example in ds:
        if max_samples and count >= max_samples:
            break

        # Get problem and solution with reasoning
        problem = example.get("problem", example.get("question", ""))
        solution = example.get("solution", example.get("response", ""))

        if not problem or not solution:
            continue

        # Include reasoning trace if available
        reasoning = example.get("reasoning", example.get("thinking", ""))
        if reasoning:
            # Format with chain-of-thought
            full_solution = f"<thinking>\n{reasoning}\n</thinking>\n\n{solution}"
        else:
            full_solution = solution

        yield {
            "messages": [
                {"role": "user", "content": f"Solve the following problem:\n\n{problem}"},
                {"role": "assistant", "content": full_solution},
            ],
            "source": "codeforces_cots",
        }
        count += 1


def load_competitive_datasets(
    streaming: bool = False,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """Load all Phase 5 competitive programming datasets.

    Args:
        streaming: Whether to stream datasets
        max_samples_per_dataset: Max samples from each dataset
        shuffle: Whether to shuffle datasets
        seed: Random seed for shuffling

    Yields:
        Dict with 'messages', 'source', and metadata
    """
    # Load all datasets
    taco_data = list(load_taco(streaming=streaming, max_samples=max_samples_per_dataset))
    codeforces = list(load_codeforces_cots(streaming=streaming, max_samples=max_samples_per_dataset))

    all_data = taco_data + codeforces

    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)

    for item in all_data:
        yield item


def get_competitive_dataset_info() -> Dict[str, Any]:
    """Get information about competitive programming datasets."""
    return {
        "phase": 5,
        "name": "Competitive Programming (Reasoning Boost)",
        "datasets": [
            {
                "name": "TACO",
                "hf_path": "BAAI/TACO",
                "size": 25433,
                "format": "problem/solutions with algorithm labels",
                "difficulties": ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"],
                "skills": [
                    "Amortized analysis", "Bit manipulation", "Complete search",
                    "Data structures", "Dynamic programming", "Greedy algorithms",
                    "Range queries", "Sorting",
                ],
            },
            {
                "name": "CodeForces-CoTs",
                "hf_path": "open-r1/codeforces-cots",
                "size": "10K problems + ~100K traces",
                "format": "problem/solution with chain-of-thought reasoning",
            },
        ],
        "total_estimated": "~35K problems",
    }
