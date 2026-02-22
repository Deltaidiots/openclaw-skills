---
name: wandb
description: Inspect, compare, and summarize Weights & Biases (W&B) runs and projects. Use when the user asks about the latest/current run, run health (loss/throughput/instability), comparing runs across hyperparameters, generating run reports, or diagnosing RL/LLM training signals (optionally including GRPO/PPO-style metrics like KL/entropy/clipfrac) from W&B.
---

# W&B run analysis workflow

## Ask for inputs (only if missing)

- `entity/project` (e.g. `gujjar19/rlvr-playground`)
- What the user wants:
  - **Latest run health** ("is it behaving well?")
  - **Compare runs** ("which config is better?")
  - **Generate report** (Good/Bad/Ugly + recommendations)
- Run selection:
  - latest N runs, OR
  - filter by `state` / `tags`, OR
  - explicit run ids
- If project is private: ensure `WANDB_API_KEY` is available in environment (do not print it).

## Outputs to produce

For any report/comparison, prefer a structured answer:

- **Good**: what’s improving / what looks healthy
- **Bad**: what limits performance (undertraining, conservative updates, instability)
- **Ugly**: clear failures (crashes, NaNs, divergence) and likely root causes
- **Next experiments**: 3–6 concrete follow-ups (ablation-style)

If this is RL/LLM training, optionally add a short RL lens:

- KL (too low = too conservative; too high/spiky = too aggressive)
- entropy (collapse = determinism/mode collapse risk)
- clipfrac/grad_norm/loss spikes
- reward vs task-metric mismatch (reward hacking / verifier issues)

## Run the bundled script

Use `scripts/wandb_report.py`.

Examples:

```bash
python3 skills/wandb/scripts/wandb_report.py \
  --entity <ENTITY> \
  --project <PROJECT> \
  --state any \
  --max-runs 80 \
  --out /tmp/wandb_report.md
```

Override metric keys if your logging uses different names:

```bash
python3 skills/wandb/scripts/wandb_report.py \
  --entity <ENTITY> --project <PROJECT> \
  --primary eval/success_rate \
  --reward train/reward_mean \
  --kl actor/ppo_kl \
  --entropy actor/entropy \
  --grad-norm actor/grad_norm \
  --tokens perf/total_num_tokens \
  --out /tmp/wandb_report.md
```

## Notes

- If the user doesn’t know the "primary metric": infer the best available task metric from W&B summary keys (prefer eval/accuracy/success/exact_match; fall back to reward/score only if needed).
- For quick "latest run" answers: pull a bounded history window (last N samples) for key metrics and summarize trends.
