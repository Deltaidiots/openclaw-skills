#!/usr/bin/env python3
"""Backward-compatible wrapper.

This repository originally shipped the script as `wandb_grpo_report.py`.
The skill was renamed to a generic `wandb` skill; the main script is now:

  - wandb_report.py

Keep this file so older docs/commands don't break.
"""

from wandb_report import main


if __name__ == "__main__":
    main()
