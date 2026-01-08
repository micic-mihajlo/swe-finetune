"""Trajectory parsing utilities for SWE-bench agent data."""

from typing import Dict, Any, List
import json


def parse_trajectory(trajectory: Any) -> List[Dict[str, Any]]:
    """Parse trajectory into list of steps."""
    if isinstance(trajectory, str):
        try:
            trajectory = json.loads(trajectory)
        except json.JSONDecodeError:
            return [{"action": trajectory}]

    if isinstance(trajectory, list):
        return trajectory
    elif isinstance(trajectory, dict):
        if "steps" in trajectory:
            return trajectory["steps"]
        return [trajectory]
    return []


def format_trajectory_as_messages(trajectory: Any) -> List[Dict[str, str]]:
    """Convert trajectory to conversation messages."""
    messages = []
    steps = parse_trajectory(trajectory)

    for step in steps:
        if isinstance(step, dict):
            if "observation" in step:
                messages.append({"role": "user", "content": step["observation"]})
            if "action" in step:
                messages.append({"role": "assistant", "content": step["action"]})
            elif "role" in step and "content" in step:
                messages.append(step)
        elif isinstance(step, str):
            role = "user" if len(messages) % 2 == 0 else "assistant"
            messages.append({"role": role, "content": step})

    return messages
