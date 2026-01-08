"""Base configuration classes for training."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 5e-5
    batch_size: int = 32
    max_length: int = 4096
    num_epochs: int = 1

    # Optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # Checkpointing
    checkpoint_every: int = 100
    eval_every: int = 500

    # LR schedule
    lr_schedule: str = "linear"  # "linear", "cosine", "constant"
    warmup_steps: int = 100


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = ""
    hf_path: str = ""
    split: str = "train"
    streaming: bool = False
    max_samples: Optional[int] = None

    # Field mappings (dataset-specific)
    instruction_field: str = "instruction"
    response_field: str = "response"
    messages_field: str = "messages"


@dataclass
class PhaseConfig:
    """Complete phase configuration."""
    name: str = ""
    description: str = ""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    datasets: List[DatasetConfig] = field(default_factory=list)

    # Google Drive paths
    checkpoint_dir: str = "/content/drive/MyDrive/swe-finetune/checkpoints"
    log_dir: str = "/content/drive/MyDrive/swe-finetune/logs"

    # Resume from checkpoint
    resume_from: Optional[str] = None

    def get_checkpoint_path(self, step: int) -> str:
        """Get checkpoint path for a given step."""
        return f"{self.checkpoint_dir}/{self.name}/step_{step:06d}"

    def get_final_path(self) -> str:
        """Get path for final model."""
        return f"{self.checkpoint_dir}/{self.name}/final"
