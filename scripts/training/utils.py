"""Training utilities for Google Colab."""

import os
import shutil
from typing import Optional
from pathlib import Path


def mount_drive(mount_path: str = "/content/drive"):
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount(mount_path)
        print(f"Drive mounted at {mount_path}")
        return True
    except ImportError:
        print("Not running in Colab, skipping drive mount")
        return False


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("step_"):
            try:
                step = int(name.split("_")[1])
                checkpoints.append((step, os.path.join(checkpoint_dir, name)))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        # Check for 'final' checkpoint
        final_path = os.path.join(checkpoint_dir, "final")
        if os.path.exists(final_path):
            return final_path
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def save_to_drive(
    local_path: str,
    drive_path: str,
    drive_mount: str = "/content/drive/MyDrive",
):
    """Copy checkpoint to Google Drive for persistence.

    Args:
        local_path: Local checkpoint path
        drive_path: Relative path within Drive
        drive_mount: Drive mount point
    """
    full_drive_path = os.path.join(drive_mount, drive_path)
    os.makedirs(os.path.dirname(full_drive_path), exist_ok=True)

    if os.path.isdir(local_path):
        if os.path.exists(full_drive_path):
            shutil.rmtree(full_drive_path)
        shutil.copytree(local_path, full_drive_path)
    else:
        shutil.copy2(local_path, full_drive_path)

    print(f"Saved to Drive: {full_drive_path}")


def load_from_drive(
    drive_path: str,
    local_path: str,
    drive_mount: str = "/content/drive/MyDrive",
) -> bool:
    """Load checkpoint from Google Drive.

    Args:
        drive_path: Relative path within Drive
        local_path: Local destination path
        drive_mount: Drive mount point

    Returns:
        True if loaded, False if not found
    """
    full_drive_path = os.path.join(drive_mount, drive_path)

    if not os.path.exists(full_drive_path):
        return False

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.isdir(full_drive_path):
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        shutil.copytree(full_drive_path, local_path)
    else:
        shutil.copy2(full_drive_path, local_path)

    print(f"Loaded from Drive: {full_drive_path}")
    return True


def get_checkpoint_dir(phase_name: str, base_dir: str = "swe-finetune/checkpoints") -> str:
    """Get checkpoint directory for a phase."""
    return os.path.join(base_dir, phase_name)


def estimate_training_time(
    num_examples: int,
    batch_size: int,
    seconds_per_step: float = 10.0,
) -> str:
    """Estimate training time.

    Args:
        num_examples: Number of training examples
        batch_size: Batch size
        seconds_per_step: Estimated seconds per training step

    Returns:
        Human-readable time estimate
    """
    num_steps = num_examples // batch_size
    total_seconds = num_steps * seconds_per_step

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return f"~{hours}h {minutes}m ({num_steps} steps)"
