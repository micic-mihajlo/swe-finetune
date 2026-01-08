# SWE-Finetune

Fine-tuning Qwen3-30B-A3B for **SWE-bench** and **TerminalBench** using [Tinker API](https://tinker-docs.thinkingmachines.ai/).

## Quick Start

1. Open `notebooks/swe_finetune.ipynb` in Google Colab
2. Set your `TINKER_API_KEY`
3. Select phase (1-5) and run

## Training Phases

| Phase | Focus | Datasets | Examples |
|-------|-------|----------|----------|
| 1 | Coding Foundation | Magicoder, Evol-Instruct | ~155K |
| 2 | Terminal/Shell | NL-SHELL-MULTI, NL2SH-ALFA | ~145K |
| 3 | Tool-Use | xLAM, Glaive function calling | ~180K |
| **4** | **SWE-bench Trajectories** | SWE-agent-trajectories, SWE-smith | **~156K** |
| 5 | Competitive Programming | TACO, CodeForces-CoT | ~35K |

**Phase 4 is critical** for SWE-bench performance.

## Structure

```
swe-finetune/
├── notebooks/
│   └── swe_finetune.ipynb    # Main training notebook
├── configs/                   # Phase configurations
├── scripts/
│   ├── data_loaders/         # HuggingFace dataset loaders
│   ├── preprocessing/        # Tokenization utilities
│   └── training/             # Training loop & utils
├── skills/tinker/            # Tinker API reference docs
└── FINETUNING_STRATEGY.md    # Detailed strategy
```

## Configuration

Each phase has its own config in `configs/`:
- `phase1_coding.py` - LR: 5e-5, Batch: 128, MaxLen: 4096
- `phase2_terminal.py` - LR: 3e-5, Batch: 64, MaxLen: 2048
- `phase3_tooluse.py` - LR: 3e-5, Batch: 32, MaxLen: 8192
- `phase4_swebench.py` - LR: 2e-5, Batch: 16, MaxLen: 16384
- `phase5_competitive.py` - LR: 3e-5, Batch: 32, MaxLen: 8192

## Expected Results

Based on similar training approaches:
- **SWE-bench Verified**: 35-45% (up from baseline)
- **TerminalBench**: Competitive performance

## License

MIT
