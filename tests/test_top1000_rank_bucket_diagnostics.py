from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from opening_strength_fit.legacy.top1000_rank_data import (
    TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
    TOP1000_RETURN_HISTOGRAM_Y_LIMITS,
    TOP1000_SCORE_BUCKETS,
    load_ranked_pool_shard,
)
from opening_strength_fit.legacy.top1000_return_histograms import (
    plot_score_bucket_histograms,
    plot_score_bucket_histograms_full_scale,
)


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


def test_ranked_pool_shard_can_reuse_embedded_next_label(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "date": ["2025-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "decision_target_timestamp": pd.to_datetime(["2025-01-02 09:31:00"] * 3),
            "prediction": [0.3, 0.1, 0.2],
            "label_next_close": [0.03, -0.01, 0.01],
        }
    ).to_parquet(pred_path, index=False)
    pool = pd.DataFrame(
        [[True, True, True]],
        index=["2025-01-02"],
        columns=["000001.SZ", "000002.SZ", "000003.SZ"],
    )

    frame, trace = load_ranked_pool_shard(
        pred_path=pred_path,
        labels=None,
        pool=pool,
        prediction_next_label_col="label_next_close",
    )

    assert frame["symbol"].tolist() == ["000001.SZ", "000003.SZ", "000002.SZ"]
    assert frame["score_rank"].tolist() == [1, 2, 3]
    assert frame["excess_bps"].tolist() == pytest.approx([200.0, 0.0, -200.0])
    assert trace["next_label_source"] == "prediction:label_next_close"
    assert trace["missing_labels"] == 0


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


def test_score_bucket_histogram_full_scale_includes_sparse_tails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histogram = _histogram()
    histogram = pd.concat(
        [
            histogram,
            pd.DataFrame(
                [
                    {
                        "score_bucket": 1,
                        "midpoint_bps": -5_350.0,
                        "observations": 1,
                    },
                    {
                        "score_bucket": 10,
                        "midpoint_bps": 5_750.0,
                        "observations": 1,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    saved: list[tuple[Figure, Path]] = []
    original_close = plt.close

    def record_savefig(figure: Figure, path: Path, **_: object) -> None:
        saved.append((figure, Path(path)))

    monkeypatch.setattr(Figure, "savefig", record_savefig)
    monkeypatch.setattr(plt, "close", lambda _: None)

    plot_score_bucket_histograms_full_scale(
        histogram,
        bin_width_bps=100,
        output_dir=tmp_path,
        variant="candidate",
    )

    figure = saved[0][0]
    axis = figure.axes[0]
    assert axis.get_xlim() == pytest.approx((-5_800.0, 5_800.0))
    assert axis.get_ylim()[0] < 1
    assert axis.get_ylim()[1] > histogram["observations"].max()
    assert [path.name for _, path in saved] == [
        "top1000_score_bucket_return_100bps_counts_full_scale.svg",
        "top1000_score_bucket_return_100bps_counts_full_scale.png",
    ]
    original_close(figure)
