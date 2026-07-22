from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

SCRIPT_PATH = (
    Path(__file__).parents[1] / "experiments" / "scripts" / "run_top1000_rank_bucket_diagnostics.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location("top1000_rank_bucket_diagnostics", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS = SCRIPT_MODULE.TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS
TOP1000_RETURN_HISTOGRAM_Y_LIMITS = SCRIPT_MODULE.TOP1000_RETURN_HISTOGRAM_Y_LIMITS
TOP1000_SCORE_BUCKETS = SCRIPT_MODULE.TOP1000_SCORE_BUCKETS
plot_score_bucket_histograms = SCRIPT_MODULE.plot_score_bucket_histograms


def _histogram() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score_bucket": bucket,
                "midpoint_bps": midpoint,
                "observations": 1_000 + bucket,
            }
            for bucket in TOP1000_SCORE_BUCKETS
            for midpoint in (-50.0, 50.0)
        ]
    )


def test_score_bucket_histogram_plot_has_fixed_acceptance_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[tuple[Figure, Path]] = []
    original_close = plt.close

    def record_savefig(figure: Figure, path: Path, **_: object) -> None:
        saved.append((figure, Path(path)))

    monkeypatch.setattr(Figure, "savefig", record_savefig)
    monkeypatch.setattr(plt, "close", lambda _: None)

    plot_score_bucket_histograms(
        _histogram(),
        bin_width_bps=100,
        output_dir=tmp_path,
        variant="candidate",
    )

    figure = saved[0][0]
    axis = figure.axes[0]
    assert axis.get_xlim() == pytest.approx(
        (-TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS, TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS)
    )
    assert axis.get_ylim() == pytest.approx(TOP1000_RETURN_HISTOGRAM_Y_LIMITS)
    assert axis.get_yscale() == "log"
    assert [path.suffix for _, path in saved] == [".svg", ".png"]
    assert len(axis.get_legend().get_texts()) == 10
    original_close(figure)


def test_score_bucket_histogram_plot_requires_all_ten_rank_buckets(tmp_path: Path) -> None:
    histogram = _histogram().loc[lambda frame: frame["score_bucket"] < 10]

    with pytest.raises(ValueError, match="must contain score buckets"):
        plot_score_bucket_histograms(
            histogram,
            bin_width_bps=100,
            output_dir=tmp_path,
            variant="candidate",
        )
