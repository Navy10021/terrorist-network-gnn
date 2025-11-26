# Experiments Directory

This directory contains experiment outputs and results.

## Structure

```
experiments/
├── phase1_full/           # Full Phase 1 experiments
│   ├── results_phase1.json
│   └── figures/
├── quick_test/            # Quick test runs
└── custom/                # Custom experiments
```

## Output Format

Each experiment creates:
- `results_phase1.json` - Complete experimental results
- `figures/` - Visualization outputs (PNG format)
- `models/` - Saved model checkpoints (if enabled)

## Usage

Results are automatically saved when running experiments:

```bash
python scripts/run_experiment.py --output-dir experiments/my_experiment
```

## .gitignore

All experiment outputs are ignored by git (except this README).
