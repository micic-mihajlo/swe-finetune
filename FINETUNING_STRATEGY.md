# Optimal Fine-Tuning Strategy for SWE-bench + TerminalBench

## Executive Summary

Based on comprehensive research across 50+ HuggingFace datasets, this document outlines a multi-phase fine-tuning strategy to maximize performance on **SWE-bench** (code editing, bug fixing) and **TerminalBench** (shell/CLI proficiency).

**Key insight from previous attempt**: The model trained on O1 tool-calling format performed poorly because of format mismatch with OpenHands scaffolding. **Format alignment is critical.**

---

## Recommended Dataset Pipeline

### Phase 1: Foundation (General Coding Competence)
**Goal**: Build strong code understanding and generation capabilities.

| Dataset | Size | Purpose |
|---------|------|---------|
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75K | Diverse, realistic coding problems from OSS |
| `nickrosh/Evol-Instruct-Code-80k-v1` | 80K | WizardCoder-style evolved instructions |
| `bigcode/commitpackft` | 2GB | Real git commits as natural instructions |

**Total**: ~155K examples + commit data
**Recommended epochs**: 1
**Batch size**: 128
**Max length**: 4096

---

### Phase 2: Terminal/Shell Mastery (TerminalBench Focus)
**Goal**: Master command-line operations, system administration, bash scripting.

| Dataset | Size | Purpose |
|---------|------|---------|
| `Mitchins/NL-SHELL-MULTI` | 78K | Combined NL2Bash + TLDR + NL2SH-ALFA |
| `westenfelder/NL2SH-ALFA` | 41K | Verified pairs with difficulty levels |
| `b-mc2/cli-commands-explained` | 16K | Commands with detailed explanations |
| `neulab/tldr` | 9K + docs | NL-to-bash with manual references |
| `harpomaxx/unix-commands` | 100 | Commands with expected outputs |

**Total**: ~145K examples
**Recommended epochs**: 2
**Batch size**: 64
**Max length**: 2048

---

### Phase 3: Tool-Use & Function Calling
**Goal**: Learn structured tool invocation patterns (critical for agent scaffolding).

| Dataset | Size | Purpose |
|---------|------|---------|
| `Salesforce/xlam-function-calling-60k` | 60K | Top function-calling dataset |
| `glaiveai/glaive-function-calling-v2` | 113K | Multi-turn function calling |
| `NousResearch/hermes-function-calling-v1` | 5K+ | Clean ShareGPT format with tool tags |
| `Team-ACE/ToolACE` | 26K APIs | Complex multi-step tool interactions |

**Total**: ~200K examples
**Recommended epochs**: 2
**Batch size**: 32
**Max length**: 8192

---

### Phase 4: SWE-bench Agent Trajectories (Critical!)
**Goal**: Learn the exact interaction patterns for code editing agents.

| Dataset | Size | Quality | Notes |
|---------|------|---------|-------|
| `nebius/SWE-agent-trajectories` | 80K | 40.6% on Verified | CC-BY-4.0, best scale |
| `SWE-bench/SWE-smith-trajectories` | **76K** | Claude 3.7 traces | Official, high quality |
| `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | 66K | 39% at 8B | Linear scaling with data |
| `R2E-Gym/R2E-Gym-V1` | 8K+ | 51% SOTA | Smaller but highest quality |

**NEW (Dec 2025)**: SWE-smith is now split into language-specific datasets:
- `SWE-bench/SWE-smith-py` - **50.9K** Python task instances
- `SWE-bench/SWE-smith-go` - **8.2K** Go task instances
- Other languages still being built (Java, JS, Rust, PHP)

**CRITICAL**: Choose trajectories that match your target scaffolding format!
- For **OpenHands**: Use `nebius/SWE-rebench-openhands-trajectories`
- For **SWE-agent**: Use `nebius/SWE-agent-trajectories`

**Total**: 80K-150K trajectories
**Recommended epochs**: 3
**Batch size**: 16-32
**Max length**: 16384 (trajectories are long!)

---

### Phase 5: Competitive Programming (Reasoning Boost)
**Goal**: Strengthen algorithmic reasoning and problem-solving.

| Dataset | Size | Purpose |
|---------|------|---------|
| `BAAI/TACO` | 25K problems | Fine-grained algorithm labels |
| `open-r1/codeforces-cots` | 10K + 100K traces | CoT reasoning traces |
| `ByteDance-Seed/Code-Contests-Plus` | 11.7K | Verified test cases |

**Total**: ~35K problems (use solutions, not just problems)
**Recommended epochs**: 1
**Batch size**: 32
**Max length**: 8192

---

## Training Configuration for Tinker API

```python
from dataclasses import dataclass

@dataclass
class Phase1Config:
    """General coding competence"""
    learning_rate: float = 5e-5
    batch_size: int = 128
    max_length: int = 4096
    num_epochs: int = 1
    datasets: list = ("Magicoder-OSS-Instruct-75K", "Evol-Instruct-Code-80k-v1")

@dataclass
class Phase2Config:
    """Terminal/shell mastery"""
    learning_rate: float = 3e-5  # Lower LR for fine-grained skills
    batch_size: int = 64
    max_length: int = 2048
    num_epochs: int = 2
    datasets: list = ("NL-SHELL-MULTI", "NL2SH-ALFA", "cli-commands-explained")

@dataclass
class Phase3Config:
    """Tool-use patterns"""
    learning_rate: float = 3e-5
    batch_size: int = 32
    max_length: int = 8192
    num_epochs: int = 2
    datasets: list = ("xlam-function-calling-60k", "glaive-function-calling-v2")

@dataclass
class Phase4Config:
    """SWE-bench trajectories - MOST IMPORTANT"""
    learning_rate: float = 2e-5  # Lower LR for long sequences
    batch_size: int = 16
    max_length: int = 16384
    num_epochs: int = 3
    datasets: list = ("SWE-agent-trajectories", "SWE-smith-trajectories")

@dataclass
class Phase5Config:
    """Competitive programming"""
    learning_rate: float = 3e-5
    batch_size: int = 32
    max_length: int = 8192
    num_epochs: int = 1
    datasets: list = ("TACO", "codeforces-cots")

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 64  # Increased from 32 for more capacity
```

---

## Format Alignment Strategy

### The Problem
Your previous fine-tune used O1's native tool-calling format:
```json
{"type": "function", "function": {"name": "...", "arguments": "..."}}
```

But OpenHands uses a different format with XML-like tags and specific action types.

### The Solution

**Option A**: Fine-tune on OpenHands-format trajectories
- Use `nebius/SWE-rebench-openhands-trajectories`
- Matches exact inference format

**Option B**: Convert trajectories to target format during preprocessing
- Parse existing SWE-agent trajectories
- Transform to OpenHands action format
- More data available but requires conversion

**Option C**: Use SWE-agent for inference (matches training data)
- Fine-tune on SWE-agent format (nebius dataset)
- Run inference with SWE-agent scaffolding instead of OpenHands

---

## Data Preprocessing Pipeline

```python
from datasets import load_dataset
from tinker import types

def load_and_process_phase(phase_config, tokenizer, renderer):
    """Generic phase processing"""
    all_data = []

    for dataset_name in phase_config.datasets:
        ds = load_dataset(dataset_name, split="train")

        for example in ds:
            # Convert to conversation format
            messages = convert_to_messages(example, dataset_name)

            # Build supervised example with proper weights
            model_input, weights = renderer.build_supervised_example(
                messages,
                max_length=phase_config.max_length
            )

            # Create Datum
            tokens = model_input.tokens[:-1]
            target_tokens = model_input.tokens[1:]
            weights = weights[1:]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=tokens),
                loss_fn_inputs=dict(
                    weights=weights,
                    target_tokens=target_tokens
                )
            )
            all_data.append(datum)

    return all_data

def convert_to_messages(example, dataset_name):
    """Convert dataset-specific format to conversation format"""

    if "instruction" in example:  # Magicoder, Evol-Instruct style
        return [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]

    elif "command" in example:  # NL2Bash style
        return [
            {"role": "user", "content": f"Write a bash command to: {example['description']}"},
            {"role": "assistant", "content": example["command"]}
        ]

    elif "trajectory" in example:  # SWE-agent trajectory
        # Multi-turn: each step is a user observation + assistant action
        messages = []
        for step in example["trajectory"]:
            messages.append({"role": "user", "content": step["observation"]})
            messages.append({"role": "assistant", "content": step["action"]})
        return messages

    # Add more converters as needed...
```

---

## Expected Results

Based on published benchmarks with similar training approaches:

| Model/Approach | SWE-bench Verified | Training Data |
|----------------|-------------------|---------------|
| Klear-Agent-8B | 39% | 66K trajectories |
| SWE-agent-LM-32B | 40.2% | 5K Claude trajectories |
| Nebius trajectories | 40.6% | 80K trajectories |
| R2E-Gym agents | 51% | 8K+ high-quality |

**Realistic target for Qwen3-30B-A3B with this strategy**: 35-45% on SWE-bench Verified

**TerminalBench target**: Top-tier performance with 145K shell examples

---

## Recommended Training Order

1. **Phase 1** (Coding foundation) - 1 epoch
2. **Phase 2** (Terminal skills) - 2 epochs
3. **Phase 3** (Tool-use) - 2 epochs
4. **Phase 4** (SWE trajectories) - 3 epochs ← **Most important**
5. **Phase 5** (Competitive programming) - 1 epoch

**Alternative**: Mixed training
- Combine all datasets with appropriate sampling ratios
- Oversample SWE trajectories (3-5x)
- Single multi-epoch training run

---

## Quick Start Commands

```bash
# Download key datasets
pip install datasets

python -c "
from datasets import load_dataset

# Phase 4 (most important)
ds = load_dataset('nebius/SWE-agent-trajectories', split='train')
print(f'SWE trajectories: {len(ds)} examples')

# Phase 2 (TerminalBench)
ds = load_dataset('Mitchins/NL-SHELL-MULTI', split='train')
print(f'Shell commands: {len(ds)} examples')

# Phase 1 (Coding)
ds = load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train')
print(f'Coding instructions: {len(ds)} examples')
"
```

---

## Key Lessons from Previous Attempt

1. **Format mismatch killed performance**: O1 format ≠ OpenHands format
2. **Condenser was necessary**: 32K context was hit
3. **Patch quality was poor**: Model attempted solutions but syntax/logic errors
4. **Low resolve rate (3.7%)**: Indicates fundamental capability gap

**This strategy addresses all issues by**:
- Using scaffolding-matched trajectory data
- Training on longer sequences (16K)
- Multi-phase curriculum for robust capabilities
- Larger LoRA rank (64) for more capacity
