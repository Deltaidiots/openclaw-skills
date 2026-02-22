#!/usr/bin/env python3
"""Fetch W&B runs and generate a GRPO-style "Good / Bad / Ugly" report.

Design goals:
- works for private projects via WANDB_API_KEY in env
- uses run.config + run.summary (fast)
- includes light GRPO/PPO heuristics (KL/entropy/reward mismatch)

Usage:
  python3 wandb_grpo_report.py --entity gujjar19 --project rlvr-playground --out /tmp/report.md
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import wandb


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


def coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = float(s)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None
    return None


@dataclass
class Keys:
    primary: str
    reward: Optional[str] = None
    kl: Optional[str] = None
    entropy: Optional[str] = None
    grad_norm: Optional[str] = None
    loss: Optional[str] = None
    tokens: Optional[str] = None


@dataclass
class Flags:
    primary_missing: bool = False
    nan_or_inf: bool = False
    likely_diverged: bool = False
    entropy_collapse: bool = False
    kl_too_low: bool = False
    reward_hacking_suspected: bool = False
    undertrained: bool = False


def pick_first(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def guess_keys(cols: List[str]) -> Keys:
    # We store summary under sum.* and config under cfg.*
    # Prefer summary keys.
    primary = pick_first(cols, [
        # Preferred: task correctness metrics
        "sum.eval/success_rate",
        "sum.eval/accuracy",
        "sum.eval/exact_match",
        "sum.eval/pass@1",
        "sum.val/accuracy",
        # veRL often logs reward/score stats instead of explicit eval metrics
        "sum.critic/score/mean",
        "sum.critic/rewards/mean",
    ]) or ("sum.critic/score/mean" if "sum.critic/score/mean" in set(cols) else cols[0])

    reward = pick_first(cols, ["sum.train/reward_mean", "sum.reward/mean", "sum.episode_reward_mean", "sum.train/reward"])
    kl = pick_first(cols, ["sum.train/kl_mean", "sum.kl/mean", "sum.approx_kl", "sum.train/kl"])
    entropy = pick_first(cols, ["sum.train/entropy", "sum.entropy", "sum.policy/entropy"])
    grad_norm = pick_first(cols, ["sum.train/grad_norm", "sum.grad_norm"])
    loss = pick_first(cols, ["sum.train/loss", "sum.loss", "sum.policy_loss"])
    tokens = pick_first(cols, ["sum.train/tokens", "sum.tokens", "sum.tokens/total"])

    return Keys(primary=primary, reward=reward, kl=kl, entropy=entropy, grad_norm=grad_norm, loss=loss, tokens=tokens)


def compute_flags(row: pd.Series, keys: Keys) -> Flags:
    f = Flags()

    prim = coerce_float(row.get(keys.primary))
    if prim is None:
        f.primary_missing = True

    for k in [keys.primary, keys.reward, keys.kl, keys.entropy, keys.grad_norm, keys.loss]:
        if not k:
            continue
        v = row.get(k)
        if isinstance(v, (float, int)):
            if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or (isinstance(v, (int, float)) and math.isinf(float(v))):
                f.nan_or_inf = True

    gn = coerce_float(row.get(keys.grad_norm)) if keys.grad_norm else None
    loss = coerce_float(row.get(keys.loss)) if keys.loss else None
    ent = coerce_float(row.get(keys.entropy)) if keys.entropy else None
    kl = coerce_float(row.get(keys.kl)) if keys.kl else None
    rew = coerce_float(row.get(keys.reward)) if keys.reward else None

    if gn is not None and gn > 1e3:
        f.likely_diverged = True
    if loss is not None and abs(loss) > 1e6:
        f.likely_diverged = True

    if ent is not None and ent < 0.1:
        f.entropy_collapse = True

    if kl is not None and kl < 0.01:
        f.kl_too_low = True

    steps = coerce_float(row.get("sum._step")) or coerce_float(row.get("_step"))
    tokens = coerce_float(row.get(keys.tokens)) if keys.tokens else None
    if steps is not None and steps < 100:
        f.undertrained = True
    if tokens is not None and tokens < 1e6:
        f.undertrained = True

    # very rough reward-hacking heuristic: reward high but primary low
    if rew is not None and prim is not None:
        if rew > 0.8 and prim < 0.2:
            f.reward_hacking_suspected = True

    return f


def add_flags(df: pd.DataFrame, keys: Keys) -> pd.DataFrame:
    flags = [compute_flags(r, keys) for _, r in df.iterrows()]
    for field in Flags.__dataclass_fields__.keys():
        df[f"flag.{field}"] = [getattr(f, field) for f in flags]
    df["flag.any"] = df[[c for c in df.columns if c.startswith("flag.") and c != "flag.any"]].any(axis=1)
    return df


def spearman_correlations(df: pd.DataFrame, target: str, features: List[str], min_pairs: int = 8) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce")
        m = (~x.isna()) & (~y.isna())
        if int(m.sum()) < min_pairs:
            continue
        rho = pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")
        rows.append({"feature": feat, "spearman_rho": float(rho), "n": int(m.sum())})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("spearman_rho", ascending=False)


def pareto_front(df: pd.DataFrame, metric: str, cost: str, higher_is_better: bool) -> pd.DataFrame:
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d[cost] = pd.to_numeric(d[cost], errors="coerce")
    d = d.dropna(subset=[metric, cost])

    mvals = d[metric].values
    cvals = d[cost].values

    if not higher_is_better:
        mvals = -mvals

    efficient = np.ones(len(d), dtype=bool)
    for i in range(len(d)):
        if not efficient[i]:
            continue
        dominates = (mvals >= mvals[i]) & (cvals <= cvals[i]) & ((mvals > mvals[i]) | (cvals < cvals[i]))
        if np.any(dominates):
            efficient[i] = False

    out = d.loc[efficient].copy()
    out = out.sort_values([cost, metric], ascending=[True, False])
    return out


def md_table(df: pd.DataFrame, cols: List[str], n: int) -> str:
    keep = [c for c in cols if c in df.columns]
    return df[keep].head(n).to_markdown(index=False)


def fetch_runs(entity: str, project: str, state: Optional[str], tags: List[str], max_runs: int) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    path = f"{entity}/{project}"

    filters: Dict[str, Any] = {}
    if state and state != "any":
        filters["state"] = state
    if tags:
        filters["tags"] = {"$in": tags}

    runs = api.runs(path, filters=filters, per_page=min(max_runs, 200))
    out: List[wandb.apis.public.Run] = []
    for r in runs:
        out.append(r)
        if len(out) >= max_runs:
            break
    return out


def runs_to_df(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    rows = []
    for r in runs:
        cfg = {k: v for k, v in dict(r.config).items() if not str(k).startswith("_")}
        summ = dict(r.summary)

        row = {
            "run.id": r.id,
            "run.name": r.name,
            "run.state": r.state,
            "run.created_at": (r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at)) if r.created_at else None,
            "run.url": r.url,
            "sum._step": summ.get("_step"),
            "sum._runtime": summ.get("_runtime"),
        }
        row.update({f"cfg.{k}": v for k, v in flatten_dict(cfg).items()})
        row.update({f"sum.{k}": v for k, v in flatten_dict(summ).items()})
        rows.append(row)

    return pd.DataFrame(rows)


def render(df: pd.DataFrame, keys: Keys, higher_is_better: bool, top_k: int, cost_key: Optional[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    d = df.copy()
    d[keys.primary] = pd.to_numeric(d[keys.primary], errors="coerce")
    valid = d.dropna(subset=[keys.primary]).copy()

    lines: List[str] = []
    lines.append(f"# W&B GRPO Report\n\nGenerated: {now}\n")
    lines.append("## Executive summary\n")
    lines.append(f"- Runs analyzed: **{len(d)}** (valid primary: **{len(valid)}**)\n")
    lines.append(f"- Primary metric: `{keys.primary}` (higher_is_better={higher_is_better})\n")

    if len(valid) > 0:
        best = valid.sort_values(keys.primary, ascending=not higher_is_better).iloc[0]
        lines.append(f"- Best run: **{best['run.name']}** ({best['run.id']}) → {keys.primary}={best[keys.primary]:.4g}\n")
        lines.append(f"  - {best.get('run.url','')}\n")

    # GOOD
    lines.append("\n## Good (what worked)\n")
    if len(valid) > 0:
        ranked = valid.sort_values(keys.primary, ascending=not higher_is_better)
        cols = ["run.name", "run.id", keys.primary]
        for k in [keys.reward, keys.kl, keys.entropy, keys.tokens]:
            if k and k in ranked.columns:
                cols.append(k)
        if cost_key and cost_key in ranked.columns and cost_key not in cols:
            cols.append(cost_key)
        cols.append("run.url")
        lines.append("Top runs:\n\n" + md_table(ranked, cols, top_k) + "\n")

        if cost_key and cost_key in ranked.columns:
            pf = pareto_front(valid, keys.primary, cost_key, higher_is_better)
            if len(pf) > 0:
                lines.append("Pareto-efficient (quality vs cost):\n\n" + md_table(pf, ["run.name", "run.id", keys.primary, cost_key, "run.url"], top_k) + "\n")

    # BAD
    lines.append("\n## Bad (what likely held performance back)\n")
    if len(valid) > 0:
        xs = valid[keys.primary].values
        lines.append(f"{keys.primary} stats: min={np.min(xs):.4g}  median={np.median(xs):.4g}  max={np.max(xs):.4g}\n")

    flag_cols = [c for c in d.columns if c.startswith("flag.") and c not in ["flag.any"]]
    if flag_cols:
        counts = d[flag_cols].sum().sort_values(ascending=False)
        lines.append("Flag counts:\n")
        for k, v in counts.items():
            if int(v) > 0:
                lines.append(f"- {k}: {int(v)}")
        lines.append("\n")

    # UGLY
    lines.append("\n## Ugly (instability / divergence / suspicious patterns)\n")
    ugly = d[d["flag.any"] == True].copy()
    if len(ugly) == 0:
        lines.append("No runs triggered the current instability heuristics.\n")
    else:
        cols = ["run.name", "run.id", keys.primary]
        for k in [keys.reward, keys.kl, keys.entropy, keys.grad_norm, keys.loss]:
            if k and k in ugly.columns:
                cols.append(k)
        cols += flag_cols[:6]
        cols.append("run.url")
        lines.append(md_table(ugly.sort_values(keys.primary, ascending=not higher_is_better), cols, min(20, top_k)) + "\n")

    # HYPERPARAM SIGNALS
    lines.append("\n## Hyperparameter signals (attribution, not proof)\n")
    cfg_cols = [c for c in d.columns if c.startswith("cfg.")]
    num_cfg = []
    for c in cfg_cols:
        if pd.to_numeric(d[c], errors="coerce").notna().sum() >= 8:
            num_cfg.append(c)

    if len(valid) >= 8 and num_cfg:
        corr = spearman_correlations(valid, keys.primary, num_cfg)
        if len(corr) > 0:
            lines.append("Top positive correlations (Spearman):\n\n" + corr.head(12).to_markdown(index=False) + "\n")
            lines.append("Top negative correlations (Spearman):\n\n" + corr.tail(12).sort_values("spearman_rho").to_markdown(index=False) + "\n")
    else:
        lines.append("Not enough numeric config fields / valid runs to compute correlations.\n")

    # INTERPRETATION
    lines.append("\n## GRPO expert interpretation checklist\n")
    lines.append(
        "- Flat task metric + near-zero KL: updates too conservative → decrease KL penalty / increase lr / increase update epochs / reduce clipping.\n"
        "- KL spikes + metric collapse: too aggressive → lower lr, increase KL penalty, increase batch, normalize advantages/rewards.\n"
        "- Early entropy collapse: brittle / mode collapse risk → keep sampling temperature, moderate KL, inspect reward shaping.\n"
        "- Reward increases but task metric doesn't: reward hacking or verifier leak → tighten verifier, log failure types, inspect samples.\n"
        "- Countdown-specific: ensure strict parsing + penalize invalid formats; track % invalid, % wrong arithmetic, % formatting failures.\n"
    )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--state", default="any", help="Run state filter (e.g. finished, failed, running). Use 'any' for no filter.")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--max-runs", type=int, default=80)
    ap.add_argument("--out", default="/tmp/wandb_grpo_report.md")

    ap.add_argument("--primary", default=None, help="Primary metric key as stored in summary, e.g. eval/accuracy")
    ap.add_argument("--reward", default=None)
    ap.add_argument("--kl", default=None)
    ap.add_argument("--entropy", default=None)
    ap.add_argument("--grad-norm", default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--tokens", default=None)

    ap.add_argument("--higher-is-better", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--cost-key", default=None, help="Cost column: e.g. sum._runtime or sum.train/tokens")

    args = ap.parse_args()

    # Private projects need WANDB_API_KEY.
    # wandb.Api() reads key from env var.

    runs = fetch_runs(args.entity, args.project, args.state, args.tags, args.max_runs)
    if not runs:
        raise SystemExit("No runs found. Check entity/project/filters.")

    df = runs_to_df(runs)

    keys = guess_keys(list(df.columns))
    if args.primary:
        keys.primary = f"sum.{args.primary}" if f"sum.{args.primary}" in df.columns else args.primary
    if args.reward:
        keys.reward = f"sum.{args.reward}" if f"sum.{args.reward}" in df.columns else args.reward
    if args.kl:
        keys.kl = f"sum.{args.kl}" if f"sum.{args.kl}" in df.columns else args.kl
    if args.entropy:
        keys.entropy = f"sum.{args.entropy}" if f"sum.{args.entropy}" in df.columns else args.entropy
    if args.grad_norm:
        keys.grad_norm = f"sum.{args.grad_norm}" if f"sum.{args.grad_norm}" in df.columns else args.grad_norm
    if args.loss:
        keys.loss = f"sum.{args.loss}" if f"sum.{args.loss}" in df.columns else args.loss
    if args.tokens:
        keys.tokens = f"sum.{args.tokens}" if f"sum.{args.tokens}" in df.columns else args.tokens

    # choose default cost
    cost_key = args.cost_key
    if cost_key is None:
        for ck in [keys.tokens, "sum._runtime"]:
            if ck and ck in df.columns:
                cost_key = ck
                break

    df = add_flags(df, keys)
    report = render(df, keys, args.higher_is_better, args.top_k, cost_key)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
