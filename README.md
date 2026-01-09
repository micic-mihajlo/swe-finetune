# SWE-Finetune

Fine-tune Qwen3-30B-A3B for SWE-bench using Tinker API.

## Setup

```bash
pip install tinker datasets transformers tqdm numpy
export TINKER_API_KEY="your-key-here"
```

## Usage

```bash
python train_swebench.py
```

The script:
1. Loads the Phase 1 (coding) checkpoint
2. Trains on SWE-bench agent trajectories (~156K examples)
3. Saves the final model
4. Tests inference

## Checkpoints

- **Phase 1 (Coding)**: `tinker://606ee7d9-e694-5c39-940d-023030fec687:train:0/weights/phase1_coding-final`
- **Phase 4 (SWE-bench)**: Run `train_swebench.py` to create

## Configuration

Edit the top of `train_swebench.py`:

```python
LEARNING_RATE = 2e-5   # Lower for fine-tuning
BATCH_SIZE = 16        # Smaller for long sequences
MAX_LENGTH = 16384     # SWE-bench traces are long
MAX_SAMPLES = None     # Set to e.g. 10000 for testing
```

## License

MIT
