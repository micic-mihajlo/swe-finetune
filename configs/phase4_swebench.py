"""Phase 4: SWE-bench Agent Trajectories (CRITICAL!)

Datasets:
- nebius/SWE-agent-trajectories (80K)
- SWE-bench/SWE-smith-trajectories (76K)

Goal: Learn the exact interaction patterns for code editing agents.
This is the most important phase for SWE-bench performance.
"""

from .base import PhaseConfig, ModelConfig, TrainingConfig, DatasetConfig

phase4_config = PhaseConfig(
    name="phase4_swebench",
    description="SWE-bench agent trajectories - CRITICAL for SWE-bench performance",

    model=ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        lora_rank=32,
    ),

    training=TrainingConfig(
        learning_rate=2e-5,  # Lower LR for long sequences
        batch_size=16,  # Smaller batch for long contexts
        max_length=16384,  # Trajectories are very long
        num_epochs=3,  # More epochs for this critical phase
        checkpoint_every=100,
        eval_every=300,
        lr_schedule="linear",
        warmup_steps=200,
    ),

    datasets=[
        DatasetConfig(
            name="swe_agent_trajectories",
            hf_path="nebius/SWE-agent-trajectories",
            split="train",
            streaming=True,  # Large dataset
            max_samples=80000,
            # Trajectory format: step-by-step agent actions
            messages_field="trajectory",
        ),
        DatasetConfig(
            name="swe_smith_trajectories",
            hf_path="SWE-bench/SWE-smith-trajectories",
            split="train",
            streaming=True,
            max_samples=76000,
            # Claude 3.7 Sonnet traces
            messages_field="trajectory",
        ),
    ],

    checkpoint_dir="/content/drive/MyDrive/swe-finetune/checkpoints",
)


# Alternative: OpenHands format trajectories
# Use this if running inference with OpenHands scaffolding
phase4_openhands_config = PhaseConfig(
    name="phase4_swebench_openhands",
    description="SWE-bench trajectories in OpenHands format",

    model=ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        lora_rank=32,
    ),

    training=TrainingConfig(
        learning_rate=2e-5,
        batch_size=16,
        max_length=16384,
        num_epochs=3,
        checkpoint_every=100,
        eval_every=300,
        lr_schedule="linear",
        warmup_steps=200,
    ),

    datasets=[
        DatasetConfig(
            name="openhands_trajectories",
            hf_path="nebius/SWE-rebench-openhands-trajectories",
            split="train",
            streaming=True,
            messages_field="trajectory",
        ),
    ],

    checkpoint_dir="/content/drive/MyDrive/swe-finetune/checkpoints",
)
