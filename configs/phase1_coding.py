"""Phase 1: General Coding Foundation

Datasets:
- ise-uiuc/Magicoder-OSS-Instruct-75K
- nickrosh/Evol-Instruct-Code-80k-v1
- bigcode/commitpackft (Python subset)

Goal: Build strong code understanding and generation capabilities.
"""

from .base import PhaseConfig, ModelConfig, TrainingConfig, DatasetConfig

phase1_config = PhaseConfig(
    name="phase1_coding",
    description="General coding foundation - Magicoder, Evol-Instruct, CommitPackFT",

    model=ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        lora_rank=32,
    ),

    training=TrainingConfig(
        learning_rate=5e-5,
        batch_size=128,
        max_length=4096,
        num_epochs=1,
        checkpoint_every=200,
        eval_every=500,
        lr_schedule="linear",
        warmup_steps=100,
    ),

    datasets=[
        DatasetConfig(
            name="magicoder",
            hf_path="ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            streaming=False,
            instruction_field="instruction",
            response_field="response",
        ),
        DatasetConfig(
            name="evol_instruct",
            hf_path="nickrosh/Evol-Instruct-Code-80k-v1",
            split="train",
            streaming=False,
            instruction_field="instruction",
            response_field="output",
        ),
        DatasetConfig(
            name="commitpackft",
            hf_path="bigcode/commitpackft",
            split="train",
            streaming=True,  # Large dataset
            max_samples=50000,  # Sample subset
            instruction_field="message",  # Commit message
            response_field="content",  # Code diff
        ),
    ],

    checkpoint_dir="/content/drive/MyDrive/swe-finetune/checkpoints",
)
