from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_A_SHARE_SYMBOL_REGEX = (
    r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$"
)


def normalize_symbols(symbols: pd.Series) -> pd.Series:
    return symbols.astype(str).str.strip().str.upper()


def load_symbol_list(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"symbols file does not exist: {path}")

    if path.suffix.lower() in {".csv", ".gz"} or "".join(path.suffixes[-2:]) == ".csv.gz":
        frame = pd.read_csv(path)
        column = "symbol" if "symbol" in frame.columns else frame.columns[0]
        values = frame[column]
    else:
        values = pd.Series(path.read_text(encoding="utf-8").splitlines())

    return set(normalize_symbols(values).loc[lambda s: s.ne("")])


def filter_symbol_universe(
    frame: pd.DataFrame,
    *,
    symbol_regex: str | None = DEFAULT_A_SHARE_SYMBOL_REGEX,
    symbols: set[str] | None = None,
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    if symbol_col not in frame.columns:
        raise SystemExit(f"missing required column: {symbol_col}")

    normalized = normalize_symbols(frame[symbol_col])
    mask = pd.Series(True, index=frame.index)
    if symbol_regex:
        mask &= normalized.str.match(symbol_regex, na=False)
    if symbols:
        mask &= normalized.isin(symbols)

    out = frame.loc[mask].copy()
    out[symbol_col] = normalized.loc[mask].to_numpy()
    return out


def universe_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, int]:
    return {
        "input_rows": int(len(before)),
        "input_symbols": int(before["symbol"].nunique()) if "symbol" in before else 0,
        "kept_rows": int(len(after)),
        "kept_symbols": int(after["symbol"].nunique()) if "symbol" in after else 0,
    }
