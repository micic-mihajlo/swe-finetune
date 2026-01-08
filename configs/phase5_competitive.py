"""Phase 5: Competitive Programming (Reasoning Boost)

Datasets:
- BAAI/TACO (25K problems)
- open-r1/codeforces-cots (10K problems + 100K reasoning traces)

Goal: Strengthen algorithmic reasoning and problem-solving.
"""

from .base import PhaseConfig, ModelConfig, TrainingConfig, DatasetConfig

phase5_config = PhaseConfig(
    name="phase5_competitive",
    description="Competitive programming for reasoning boost",

    model=ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        lora_rank=32,
    ),

    training=TrainingConfig(
        learning_rate=3e-5,
        batch_size=32,
        max_length=8192,  # Problems + solutions can be long
        num_epochs=1,
        checkpoint_every=200,
        eval_every=500,
        lr_schedule="linear",
        warmup_steps=100,
    ),

    datasets=[
        DatasetConfig(
            name="taco",
            hf_path="BAAI/TACO",
            split="train",
            streaming=False,
            instruction_field="question",
            response_field="solutions",  # Multiple solutions available
        ),
        DatasetConfig(
            name="codeforces_cots",
            hf_path="open-r1/codeforces-cots",
            split="train",
            streaming=False,
            # Chain-of-thought reasoning traces
            instruction_field="problem",
            response_field="solution",
        ),
    ],

    checkpoint_dir="/content/drive/MyDrive/swe-finetune/checkpoints",
)
