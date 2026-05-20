from __future__ import annotations

import pandas as pd


BASE_YEARLY_COLUMNS = (
    "year",
    "train_window",
    "test_window",
    "train_rows",
    "test_rows",
    "features",
)


def preferred_metric_column(
    df: pd.DataFrame,
    preferred: str,
    fallback: str,
) -> str:
    if preferred in df.columns and df[preferred].notna().any():
        return preferred
    if fallback in df.columns:
        return fallback
    return preferred


def dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": len(df),
        "columns": len(df.columns),
    }
    if "date" in df.columns:
        summary["date_min"] = df["date"].min()
        summary["date_max"] = df["date"].max()
        summary["n_dates"] = int(df["date"].nunique())
    if "symbol" in df.columns:
        summary["n_symbols"] = int(df["symbol"].nunique())
    if "timestamp" in df.columns and len(df):
        summary["time_min"] = df["timestamp"].dt.strftime("%H:%M:%S").min()
        summary["time_max"] = df["timestamp"].dt.strftime("%H:%M:%S").max()
    if "label" in df.columns:
        summary["non_null_labels"] = int(df["label"].notna().sum())
        summary["valid_labels"] = (
            int(df["valid_label"].sum()) if "valid_label" in df.columns else None
        )
        summary["label_nan_rate"] = float(df["label"].isna().mean())
    return summary


def print_mapping(title: str, mapping: dict[str, object]) -> None:
    print(f"\n{title}:")
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def build_yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    rank_ic_mean_col = preferred_metric_column(
        df,
        "group_rank_ic_mean",
        "daily_rank_ic_mean",
    )
    rank_ic_std_col = preferred_metric_column(
        df,
        "group_rank_ic_std",
        "daily_rank_ic_std",
    )
    rank_ic_ir_col = preferred_metric_column(
        df,
        "group_rank_ic_ir",
        "daily_rank_ic_ir",
    )
    ic_mean_col = preferred_metric_column(df, "group_ic_mean", "daily_ic_mean")
    ic_ir_col = preferred_metric_column(df, "group_ic_ir", "daily_ic_ir")
    table = pd.DataFrame(
        {
            "year": df["test_year"].astype(int),
            "train_window": df["train_start_date"].astype(str)
            + " -> "
            + df["train_end_date"].astype(str),
            "test_window": df["test_start_date"].astype(str)
            + " -> "
            + df["test_end_date"].astype(str),
            "train_rows": df["train_rows"].astype(int),
            "test_rows": df["test_rows"].astype(int),
            "features": df["features"].astype(int),
            "model_r2": df["model_test_r2"],
            "rank_ic_mean": df[rank_ic_mean_col],
            "rank_ic_std": df[rank_ic_std_col],
            "rank_ic_ir": df[rank_ic_ir_col],
            "ic_mean": df[ic_mean_col],
            "ic_ir": df[ic_ir_col],
            "pooled_rank_ic": df["overall_rank_ic"],
        }
    )
    if "test_dates" in df.columns:
        table.insert(5, "test_dates", df["test_dates"].astype(int))
    elif "dates" in df.columns:
        table.insert(5, "test_dates", df["dates"].astype(int))
    if "symbols" in df.columns:
        table.insert(6, "symbols", df["symbols"].astype(int))
    if "selection_mode" in df.columns:
        table.insert(7, "selection", df["selection_mode"].astype(str))

    ordered = [column for column in BASE_YEARLY_COLUMNS if column in table.columns]
    ordered += [
        column
        for column in ("test_dates", "symbols", "selection")
        if column in table.columns
    ]
    ordered += [
        "model_r2",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_ir",
        "ic_mean",
        "ic_ir",
        "pooled_rank_ic",
    ]
    return table[ordered]
