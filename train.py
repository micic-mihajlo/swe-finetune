#!/usr/bin/env python3
"""
SWE-bench Fine-tuning Script

Two-phase training for Qwen3-30B-A3B:
  Phase 1: Coding foundation (Magicoder, Evol-Instruct)
  Phase 4: SWE-bench agent trajectories

Usage:
  python train.py --phase 1              # Start fresh, train on coding
  python train.py --phase 4              # Load Phase 1, train on SWE-bench
  python train.py --phase 1 --max 10000  # Quick test with 10K samples
"""

import argparse
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

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
LORA_RANK = 32

# Phase 1: Coding Foundation
PHASE1_CONFIG = {
    "name": "phase1_coding",
    "learning_rate": 5e-5,
    "batch_size": 128,
    "max_length": 4096,
    "checkpoint": None,  # Start fresh
}

# Phase 4: SWE-bench Trajectories
PHASE4_CONFIG = {
    "name": "phase4_swebench",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "max_length": 16384,
    "checkpoint": None,  # Set after Phase 1 completes
}

# ============================================================================
# PHASE 1: CODING DATA
# ============================================================================

def load_coding_data(max_samples=None):
    """Load Magicoder and Evol-Instruct datasets."""
    all_data = []

    # Dataset 1: Magicoder-OSS-Instruct-75K
    # Columns: problem, solution, lang, etc.
    print("Loading Magicoder...")
    try:
        ds1 = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
        count = 0
        for example in tqdm(ds1, desc="Magicoder"):
            if max_samples and count >= max_samples // 2:
                break

            problem = example.get("problem") or example.get("instruction", "")
            solution = example.get("solution") or example.get("response", "")

            if not problem or not solution:
                continue

            all_data.append({
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
                "source": "magicoder"
            })
            count += 1
    except Exception as e:
        print(f"Warning: Could not load Magicoder: {e}")

    # Dataset 2: Evol-Instruct-Code-80k
    # Columns: instruction, output
    print("Loading Evol-Instruct...")
    try:
        ds2 = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        count = 0
        for example in tqdm(ds2, desc="Evol-Instruct"):
            if max_samples and count >= max_samples // 2:
                break

            instruction = example.get("instruction", "")
            output = example.get("output", "")

            if not instruction or not output:
                continue

            all_data.append({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "source": "evol_instruct"
            })
            count += 1
    except Exception as e:
        print(f"Warning: Could not load Evol-Instruct: {e}")

    random.shuffle(all_data)
    print(f"Loaded {len(all_data)} total coding examples")
    return all_data


# ============================================================================
# PHASE 4: SWE-BENCH DATA
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

        if role in ["user", "assistant", "system"]:
            messages.append({"role": role, "content": text})

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

    # Dataset 1: nebius/SWE-agent-trajectories (80K)
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

            all_data.append({"messages": messages, "source": "swe_agent"})
            count += 1
    except Exception as e:
        print(f"Warning: Could not load SWE-agent trajectories: {e}")

    # Dataset 2: SWE-bench/SWE-smith-trajectories (76K)
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

            all_data.append({"messages": messages, "source": "swe_smith"})
            count += 1
    except Exception as e:
        print(f"Warning: Could not load SWE-smith trajectories: {e}")

    random.shuffle(all_data)
    print(f"Loaded {len(all_data)} total trajectories")
    return all_data


# ============================================================================
# TOKENIZATION
# ============================================================================

def messages_to_datum(messages, tokenizer, max_length, train_on_last_only=False):
    """Convert messages to Tinker Datum."""
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

    weights = [0.0] * (len(tokens) - 1)
    text_decoded = tokenizer.decode(tokens)

    if train_on_last_only:
        # Train on LAST assistant message only
        last_pos = text_decoded.rfind("<|im_start|>assistant")
        if last_pos != -1:
            prefix_tokens = len(tokenizer.encode(text_decoded[:last_pos + len("<|im_start|>assistant")]))
            for i in range(min(prefix_tokens, len(weights)), len(weights)):
                weights[i] = 1.0
    else:
        # Train on ALL assistant messages
        in_assistant = False
        pos = 0
        for i, tok in enumerate(tokens[:-1]):
            tok_text = tokenizer.decode([tok])
            pos += len(tok_text)
            chunk = text_decoded[max(0, pos-50):pos]
            if "<|im_start|>assistant" in chunk:
                in_assistant = True
            elif "<|im_end|>" in tok_text or "<|im_start|>user" in chunk or "<|im_start|>system" in chunk:
                in_assistant = False
            if in_assistant:
                weights[i] = 1.0

    # Fallback
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
# TRAINING
# ============================================================================

def train_phase(phase: int, max_samples: int = None, checkpoint: str = None):
    """Train a single phase."""

    if phase == 1:
        config = PHASE1_CONFIG
        data_loader = load_coding_data
        train_on_last = False
    elif phase == 4:
        config = PHASE4_CONFIG
        data_loader = load_swebench_data
        train_on_last = True
    else:
        raise ValueError(f"Unknown phase: {phase}")

    print("=" * 60)
    print(f"PHASE {phase}: {config['name']}")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load data
    print(f"\nLoading data...")
    data = data_loader(max_samples=max_samples)

    if not data:
        print("ERROR: No data loaded!")
        return None

    # Convert to datums
    print("\nConverting to datums...")
    datums = []
    for example in tqdm(data, desc="Tokenizing"):
        messages = example.get("messages", [])
        if messages:
            datum = messages_to_datum(messages, tokenizer, config["max_length"], train_on_last)
            if datum:
                datums.append(datum)

    print(f"Created {len(datums)} datums")

    if not datums:
        print("ERROR: No valid datums!")
        return None

    # Create training client
    print(f"\nSetting up training client...")
    service_client = tinker.ServiceClient()

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        training_client = service_client.create_training_client_from_state_with_optimizer(checkpoint)
    else:
        print(f"Creating new LoRA adapter (rank={LORA_RANK})")
        training_client = service_client.create_lora_training_client(
            base_model=MODEL_NAME,
            rank=LORA_RANK,
        )

    print("Training client ready!")

    # Training loop
    n_batches = max(1, len(datums) // config["batch_size"])
    print(f"\nTraining: {n_batches} batches of {config['batch_size']}")

    random.shuffle(datums)
    losses = []

    pbar = tqdm(range(n_batches), desc="Training")
    for step in pbar:
        batch_start = step * config["batch_size"]
        batch = datums[batch_start:batch_start + config["batch_size"]]

        if not batch:
            continue

        # Linear LR decay
        lr_mult = max(0.0, 1.0 - step / n_batches)
        current_lr = config["learning_rate"] * lr_mult

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
            save_result = training_client.save_state(name=f"{config['name']}-{step:06d}").result()
            print(f"\nCheckpoint: {save_result.path}")

    # Save final
    print("\nSaving final model...")
    final_save = training_client.save_state(name=f"{config['name']}-final").result()
    sampler_save = training_client.save_weights_for_sampler(name=f"{config['name']}-sampler").result()

    print(f"\n{'=' * 60}")
    print(f"PHASE {phase} COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Avg NLL: {np.mean(losses):.4f}")
    print(f"Final state: {final_save.path}")
    print(f"Sampler path: {sampler_save.path}")

    # Test inference
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    sampling_client = service_client.create_sampling_client(model_path=sampler_save.path)

    if phase == 1:
        test_prompt = "Write a Python function to check if a string is a palindrome."
    else:
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

    return final_save.path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SWE-bench Fine-tuning")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 4],
                        help="Training phase: 1 (coding) or 4 (SWE-bench)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max samples (for testing)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to resume from (required for phase 4)")
    args = parser.parse_args()

    # Phase 4 requires Phase 1 checkpoint
    if args.phase == 4 and not args.checkpoint:
        print("ERROR: Phase 4 requires --checkpoint (Phase 1 final checkpoint)")
        print("Example: python train.py --phase 4 --checkpoint 'tinker://...'")
        return

    train_phase(
        phase=args.phase,
        max_samples=args.max,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
