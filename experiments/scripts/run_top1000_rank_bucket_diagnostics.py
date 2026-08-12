from __future__ import annotations

# ruff: noqa: I001

from opening_strength_fit.legacy.top1000_rank_data import *  # noqa: F403
from opening_strength_fit.legacy.top1000_return_histograms import *  # noqa: F403
from opening_strength_fit.legacy.top1000_rank_bucket_diagnostics import *  # noqa: F403
from opening_strength_fit.legacy.top1000_rank_bucket_diagnostics import main

__all__ = sorted(name for name in globals() if not name.startswith("_"))

if __name__ == "__main__":
    main()
