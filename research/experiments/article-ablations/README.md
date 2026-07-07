# Article Ablations

This directory contains the local critical-ablation suite for the VEREDA
research documents. The suite turns each critical article cluster into a
paired experiment with explicit variants, seeds, metrics, figures, and a
"what we learned" section.

Dry run:

```bash
python3 research/experiments/article-ablations/run.py --suite local-critical --seeds 11,22,33 --steps 50 --dry-run
```

Full local run:

```bash
python3 research/experiments/article-ablations/run.py --suite local-critical --seeds 11,22,33 --steps 50
```

Raw checkpoints and logs are written under `runs/article-ablations/`. Summary
JSON, figures, and Markdown dossiers are written here.
