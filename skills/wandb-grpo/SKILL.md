---
name: wandb-grpo
description: Analyze Weights & Biases (W&B) runs for RL/LLM training with GRPO (and related PPO-style signals). Use when the user asks to inspect the latest/current run, compare multiple runs, diagnose GRPO training behavior (KL, entropy, collapse, divergence, reward hacking), or recommend hyperparameter updates for a best-performing experiment config (e.g., Countdown task).
---

# W&B GRPO analysis workflow

## Inputs to ask for (only if missing)

- `entity` and `project` (e.g., `gujjar19/rlvr-playground`)
- Primary task metric key (e.g., `eval/accuracy`, `eval/success_rate`, `eval/pass@1`) and whether higher is better
- Which runs to analyze:
  - latest N finished, OR
  - tags, OR
  - a list of run ids/names
- If project is private: ensure `WANDB_API_KEY` is set in environment (prefer `.env` in workspace).

## Produce outputs

Generate a Markdown report with:

- **Good**: best runs, what patterns look healthy, Pareto (quality vs cost) if cost metric exists
- **Bad**: what’s limiting performance + likely causes (conservative updates, too-aggressive updates, insufficient exploration)
- **Ugly**: instability, NaNs, entropy collapse, KL blow-ups, suspicious reward/metric mismatch
- **Hyperparam signals** (attribution, not proof): correlations + binned effects for numeric `config` params
- **Recommendations**: 3–6 concrete next experiments (ablation-style) + a suggested "best default" config

## Run the bundled script

Use the script `scripts/wandb_grpo_report.py`.

Typical call:

```bash
python3 skills/wandb-grpo/scripts/wandb_grpo_report.py \
  --entity <ENTITY> \
  --project <PROJECT> \
  --max-runs 80 \
  --state finished \
  --out /tmp/wandb_grpo_report.md
```

Override metric keys if logs use different names:

```bash
python3 skills/wandb-grpo/scripts/wandb_grpo_report.py \
  --entity <ENTITY> --project <PROJECT> \
  --primary eval/accuracy \
  --reward train/reward_mean \
  --kl train/kl_mean \
  --entropy train/entropy \
  --grad-norm train/grad_norm \
  --tokens train/tokens \
  --out /tmp/wandb_grpo_report.md
```

## Interpretation guardrails (GRPO-specific)

- Diagnose *direction* before tuning:
  - flat metric + near-zero KL → too conservative
  - spikes/instability + high KL → too aggressive
  - entropy → collapsing too early often predicts brittle behavior
  - reward ↑ but task metric ↔/↓ → reward hacking / verifier leak
- For Countdown tasks, watch for formatting/parsing exploits; ensure verifier is strict and log failure modes.

## If the user wants "latest run health"

- Pull last N steps of history for key metrics (KL, entropy, reward, primary metric) and summarize trends.
- Flag:
  - monotonic KL drift upward
  - entropy collapse
  - reward/metric decoupling
  - sudden loss/grad-norm spikes

(If history fetching is needed and not implemented yet, extend the script to fetch a bounded history window.)
