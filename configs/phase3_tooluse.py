"""Phase 3: Tool-Use & Function Calling

Datasets:
- Salesforce/xlam-function-calling-60k (60K)
- glaiveai/glaive-function-calling-v2 (113K)
- NousResearch/hermes-function-calling-v1

Goal: Learn structured tool invocation patterns critical for agent scaffolding.
"""

from .base import PhaseConfig, ModelConfig, TrainingConfig, DatasetConfig

phase3_config = PhaseConfig(
    name="phase3_tooluse",
    description="Tool-use and function calling patterns",

    model=ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        lora_rank=32,
    ),

    training=TrainingConfig(
        learning_rate=3e-5,
        batch_size=32,
        max_length=8192,  # Tool calls can be long
        num_epochs=2,
        checkpoint_every=200,
        eval_every=500,
        lr_schedule="linear",
        warmup_steps=100,
    ),

    datasets=[
        DatasetConfig(
            name="xlam_function_calling",
            hf_path="Salesforce/xlam-function-calling-60k",
            split="train",
            streaming=False,
            messages_field="conversations",  # Uses conversation format
        ),
        DatasetConfig(
            name="glaive_function_calling",
            hf_path="glaiveai/glaive-function-calling-v2",
            split="train",
            streaming=False,
            # Custom format: SYSTEM/USER/ASSISTANT with <functioncall>
            messages_field="conversations",
        ),
        DatasetConfig(
            name="hermes_function_calling",
            hf_path="NousResearch/hermes-function-calling-v1",
            split="train",
            streaming=False,
            # ShareGPT format with <tool_call> tags
            messages_field="conversations",
        ),
    ],

    checkpoint_dir="/content/drive/MyDrive/swe-finetune/checkpoints",
)
