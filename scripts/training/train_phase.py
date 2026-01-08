"""Generic phase training loop for Tinker."""

from typing import Iterator, Dict, Any, Optional
import tinker
from tinker import types


def create_training_client(
    model_name: str = "Qwen/Qwen3-30B-A3B",
    lora_rank: int = 32,
    seed: int = 42,
):
    """Create Tinker LoRA training client."""
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
        train_attn=True,
        train_mlp=True,
        seed=seed,
    )
    return training_client


def train_phase(
    training_client,
    data_iterator: Iterator[Dict[str, Any]],
    renderer,
    config: Dict[str, Any],
    checkpoint_callback=None,
):
    """Run training for a phase.

    Args:
        training_client: Tinker TrainingClient
        data_iterator: Iterator yielding examples with 'messages' key
        renderer: Tinker renderer for tokenization
        config: Training config dict with keys:
            - learning_rate: float
            - batch_size: int
            - num_steps: int (optional, derived from data if not set)
            - checkpoint_every: int
            - lr_schedule: str ("linear", "cosine", "constant")
            - warmup_steps: int
        checkpoint_callback: Optional callback(step, path) after each checkpoint

    Returns:
        Dict with training stats
    """
    from scripts.preprocessing import batch_to_data

    lr = config.get("learning_rate", 5e-5)
    batch_size = config.get("batch_size", 32)
    checkpoint_every = config.get("checkpoint_every", 100)
    lr_schedule = config.get("lr_schedule", "linear")
    warmup_steps = config.get("warmup_steps", 100)
    num_steps = config.get("num_steps", None)
    train_on_what = config.get("train_on_what", "all")

    # Collect batches
    batch = []
    step = 0
    total_loss = 0.0
    losses = []

    for example in data_iterator:
        batch.append(example)

        if len(batch) >= batch_size:
            # Convert to Datum
            data = batch_to_data(batch, renderer, train_on_what)
            if not data:
                batch = []
                continue

            # Calculate learning rate
            if lr_schedule == "linear" and num_steps:
                current_lr = lr * (1 - step / num_steps)
            elif lr_schedule == "cosine" and num_steps:
                import math
                current_lr = lr * 0.5 * (1 + math.cos(math.pi * step / num_steps))
            else:
                current_lr = lr

            # Warmup
            if step < warmup_steps:
                current_lr = lr * (step / warmup_steps)

            # Training step
            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            fwd_bwd = training_client.forward_backward(data, "cross_entropy")
            optim = training_client.optim_step(adam_params)

            result = fwd_bwd.result()
            optim.result()

            loss = result.loss
            total_loss += loss
            losses.append(loss)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss:.4f}, LR: {current_lr:.2e}")

            # Checkpoint
            if checkpoint_every and step > 0 and step % checkpoint_every == 0:
                path = training_client.save_state(name=f"step_{step:06d}").result().path
                print(f"Checkpoint saved: {path}")
                if checkpoint_callback:
                    checkpoint_callback(step, path)

            batch = []
            step += 1

            if num_steps and step >= num_steps:
                break

    return {
        "steps": step,
        "avg_loss": total_loss / max(step, 1),
        "losses": losses,
    }
