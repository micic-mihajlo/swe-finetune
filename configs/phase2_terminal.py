"""Phase 2: Terminal/Shell Mastery (TerminalBench Focus)

Datasets:
- Mitchins/NL-SHELL-MULTI (78K)
- westenfelder/NL2SH-ALFA (41K)
- b-mc2/cli-commands-explained (16K)
- neulab/tldr (9K)

Goal: Master command-line operations, system administration, bash scripting.
"""

from .base import PhaseConfig, ModelConfig, TrainingConfig, DatasetConfig

phase2_config = PhaseConfig(
    name="phase2_terminal",
    description="Terminal/shell mastery for TerminalBench",

    model=ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        lora_rank=32,
    ),

    training=TrainingConfig(
        learning_rate=3e-5,  # Lower LR for fine-grained skills
        batch_size=64,
        max_length=2048,  # Shell commands are shorter
        num_epochs=2,
        checkpoint_every=200,
        eval_every=500,
        lr_schedule="linear",
        warmup_steps=50,
    ),

    datasets=[
        DatasetConfig(
            name="nl_shell_multi",
            hf_path="Mitchins/NL-SHELL-MULTI",
            split="train",
            streaming=False,
            instruction_field="description",
            response_field="command",
        ),
        DatasetConfig(
            name="nl2sh_alfa",
            hf_path="westenfelder/NL2SH-ALFA",
            split="train",
            streaming=False,
            instruction_field="nl",
            response_field="bash",
        ),
        DatasetConfig(
            name="cli_commands",
            hf_path="b-mc2/cli-commands-explained",
            split="train",
            streaming=False,
            instruction_field="explanation",
            response_field="command",
        ),
        DatasetConfig(
            name="tldr",
            hf_path="neulab/tldr",
            split="train",
            streaming=False,
            instruction_field="query",
            response_field="answer",
        ),
    ],

    checkpoint_dir="/content/drive/MyDrive/swe-finetune/checkpoints",
)
