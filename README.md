# SWE-Finetune

Fine-tune Qwen3-30B-A3B for SWE-bench using [Tinker API](https://tinker-docs.thinkingmachines.ai/).

## Overview

Two-phase training approach:

1. **Phase 1: Coding Foundation** (~155K examples)
   - Magicoder-OSS-Instruct-75K
   - Evol-Instruct-Code-80K

2. **Phase 2: SWE-bench Trajectories** (~156K examples)
   - nebius/SWE-agent-trajectories (80K)
   - SWE-bench/SWE-smith-trajectories (76K)

Phase 1 builds general coding skills. Phase 2 teaches the model how to act as a software engineering agent solving real GitHub issues.

## Setup

```bash
pip install tinker datasets transformers tqdm numpy
export TINKER_API_KEY="your-key-here"
```

## Usage

### Phase 1: Coding Foundation

```bash
python train.py --phase 1
```

This creates a checkpoint like:
```
tinker://abc123.../weights/phase1_coding-final
```

### Phase 2: SWE-bench Trajectories

```bash
python train.py --phase 2 --checkpoint "tinker://abc123.../weights/phase1_coding-final"
```

### Quick Testing

```bash
python train.py --phase 1 --max 10000  # Only 10K samples
```

## Checkpoints

My Phase 1 checkpoint: `tinker://606ee7d9-e694-5c39-940d-023030fec687:train:0/weights/phase1_coding-final`

## Configuration

Edit the top of `train.py`:

```python
PHASE1_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 128,
    "max_length": 4096,
}

PHASE2_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "max_length": 16384,
}
```

## Results

Phase 1 alone gives solid coding ability. Phase 2 adds SWE-bench-specific agent behavior.

Test after training:
```
Prompt: Write a bash command to find all Python files larger than 1MB
Response: find . -type f -name "*.py" -size +1M
```

## License

MIT
