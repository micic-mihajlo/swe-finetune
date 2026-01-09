#!/usr/bin/env python3
"""
SWE-bench Fine-tuning Script

Loads Phase 1 checkpoint and trains on SWE-bench agent trajectories.
Run with: python train_swebench.py
"""

import os
import json
import random
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your Phase 1 checkpoint
PHASE1_CHECKPOINT = "tinker://606ee7d9-e694-5c39-940d-023030fec687:train:0/weights/phase1_coding-final"

# Model
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
LORA_RANK = 32

# Training
LEARNING_RATE = 2e-5  # Lower LR for fine-tuning on top of Phase 1
BATCH_SIZE = 16       # Smaller batch for longer sequences
MAX_LENGTH = 16384    # SWE-bench traces can be long
MAX_SAMPLES = None    # Set to e.g. 10000 for faster testing

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_swe_agent_trajectory(trajectory):
    """Parse nebius/SWE-agent-trajectories format.

    Each step has: role, text, mask, system_prompt, cutoff_date
    """
    messages = []

    if not isinstance(trajectory, list):
        return messages

    for step in trajectory:
        if not isinstance(step, dict):
            continue

        role = step.get("role", "")
        text = step.get("text", "")

        if not text:
            continue

        # Map roles
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text})
        elif role == "system":
            messages.append({"role": "system", "content": text})

    return messages


def parse_swe_smith_messages(messages_str):
    """Parse SWE-bench/SWE-smith-trajectories format.

    messages is a JSON string containing conversation.
    """
    if not messages_str:
        return []

    try:
        messages = json.loads(messages_str)
        if isinstance(messages, list):
            # Validate format
            result = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    result.append({"role": msg["role"], "content": msg["content"]})
            return result
    except json.JSONDecodeError:
        pass

    return []


def load_swebench_data(max_samples=None):
    """Load SWE-bench trajectory datasets."""
    all_data = []

    # Dataset 1: nebius/SWE-agent-trajectories
    # Columns: instance_id, model_name, target, trajectory (list of {role, text, ...}), exit_status, generated_patch, eval_logs
    print("Loading SWE-agent trajectories...")
    try:
        ds1 = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
        count = 0
        for example in tqdm(ds1, desc="SWE-agent"):
            if max_samples and count >= max_samples // 2:
                break

            trajectory = example.get("trajectory", [])
            messages = parse_swe_agent_trajectory(trajectory)

            if not messages or len(messages) < 2:
                continue

            all_data.append({"messages": messages, "source": "swe_agent", "instance_id": example.get("instance_id", "")})
            count += 1
    except Exception as e:
        print(f"Warning: Could not load SWE-agent trajectories: {e}")

    # Dataset 2: SWE-bench/SWE-smith-trajectories
    # Columns: split, messages (JSON string), instance_id, resolved, model, traj_id, patch
    print("Loading SWE-smith trajectories...")
    try:
        ds2 = load_dataset("SWE-bench/SWE-smith-trajectories", split="train", streaming=True)
        count = 0
        for example in tqdm(ds2, desc="SWE-smith"):
            if max_samples and count >= max_samples // 2:
                break

            messages_str = example.get("messages", "")
            messages = parse_swe_smith_messages(messages_str)

            if not messages or len(messages) < 2:
                continue

            all_data.append({"messages": messages, "source": "swe_smith", "instance_id": example.get("instance_id", "")})
            count += 1
    except Exception as e:
        print(f"Warning: Could not load SWE-smith trajectories: {e}")

    random.shuffle(all_data)
    print(f"Loaded {len(all_data)} total trajectories")
    return all_data


# ============================================================================
# TOKENIZATION
# ============================================================================

def messages_to_datum(messages, tokenizer, max_length):
    """Convert messages to Tinker Datum. Train on last assistant response only."""
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        text = "\n".join(parts)

    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    if len(tokens) < 2:
        return None

    # Train on LAST assistant message only (for SWE-bench)
    weights = [0.0] * (len(tokens) - 1)
    text_decoded = tokenizer.decode(tokens)

    last_assistant_pos = text_decoded.rfind("<|im_start|>assistant")
    if last_assistant_pos != -1:
        prefix_tokens = len(tokenizer.encode(text_decoded[:last_assistant_pos + len("<|im_start|>assistant")]))
        for i in range(min(prefix_tokens, len(weights)), len(weights)):
            weights[i] = 1.0
    else:
        # Fallback
        for i in range(len(weights) // 2, len(weights)):
            weights[i] = 1.0

    if sum(weights) == 0:
        for i in range(len(weights) // 4, len(weights)):
            weights[i] = 1.0

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.array(target_tokens, dtype=np.int64)),
            "weights": TensorData.from_numpy(np.array(weights, dtype=np.float32)),
        }
    )


def compute_mean_nll(logprobs, weights):
    """Compute mean negative log-likelihood."""
    total_loss = 0.0
    total_weight = 0.0
    for lp, w in zip(logprobs, weights):
        lp_arr = lp.to_numpy() if hasattr(lp, 'to_numpy') else np.array(lp)
        w_arr = w.to_numpy() if hasattr(w, 'to_numpy') else np.array(w)
        total_loss += float(np.sum(-lp_arr * w_arr))
        total_weight += float(np.sum(w_arr))
    return total_loss / total_weight if total_weight > 0 else 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("SWE-bench Fine-tuning (Phase 4)")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load data
    print("\nLoading SWE-bench trajectories...")
    data = load_swebench_data(max_samples=MAX_SAMPLES)

    if not data:
        print("ERROR: No data loaded!")
        return

    # Convert to datums
    print("\nConverting to datums...")
    datums = []
    for example in tqdm(data, desc="Tokenizing"):
        messages = example.get("messages", [])
        if messages:
            datum = messages_to_datum(messages, tokenizer, MAX_LENGTH)
            if datum:
                datums.append(datum)

    print(f"Created {len(datums)} datums")

    if not datums:
        print("ERROR: No valid datums!")
        return

    # Create training client from Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint...")
    print(f"  {PHASE1_CHECKPOINT}")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_training_client_from_state_with_optimizer(PHASE1_CHECKPOINT)
    print("Training client ready!")

    # Training
    n_batches = max(1, len(datums) // BATCH_SIZE)
    print(f"\nTraining: {n_batches} batches of {BATCH_SIZE}")

    random.shuffle(datums)
    losses = []

    pbar = tqdm(range(n_batches), desc="Training")
    for step in pbar:
        batch_start = step * BATCH_SIZE
        batch = datums[batch_start:batch_start + BATCH_SIZE]

        if not batch:
            continue

        # Linear LR decay
        lr_mult = max(0.0, 1.0 - step / n_batches)
        current_lr = LEARNING_RATE * lr_mult

        adam_params = types.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )

        # Forward-backward
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)
        losses.append(train_nll)

        pbar.set_postfix({"NLL": f"{train_nll:.4f}", "LR": f"{current_lr:.2e}"})

        # Checkpoint every 200 steps
        if step > 0 and step % 200 == 0:
            save_result = training_client.save_state(name=f"swebench-{step:06d}").result()
            print(f"\nCheckpoint: {save_result.path}")

    # Save final
    print("\nSaving final model...")
    final_save = training_client.save_state(name="swebench-final").result()
    sampler_save = training_client.save_weights_for_sampler(name="swebench-sampler").result()

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Avg NLL: {np.mean(losses):.4f}")
    print(f"Final state: {final_save.path}")
    print(f"Sampler path: {sampler_save.path}")

    # Test inference
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    sampling_client = service_client.create_sampling_client(model_path=sampler_save.path)

    test_prompt = "Write a Python function to find all files in a directory that were modified in the last 24 hours."
    messages = [{"role": "user", "content": test_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text)

    result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=500, temperature=0.7)
    ).result()

    response = tokenizer.decode(result.sequences[0].tokens)
    print(f"\nPrompt: {test_prompt}")
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
