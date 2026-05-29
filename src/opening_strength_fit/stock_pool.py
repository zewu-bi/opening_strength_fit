from __future__ import annotations

from dataclasses import dataclass
import base64
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opening_strength_fit.config import config_bool, config_int, config_str
from opening_strength_fit.universe import normalize_symbols


CEPH_ENDPOINTS = {
    "ssd": "http://ceph-s3-ssd.prod.highfortfunds.com",
    "hdd": "http://ceph-s3.prod.highfortfunds.com",
}

DEFAULT_STOCK_POOL_PATHS = {
    "L": "lml.bzw@ssd/data/pool_L.parquet",
    "M": "lml.bzw@ssd/data/pool_M.parquet",
    "S": "lml.bzw@ssd/data/pool_S.parquet",
}


@dataclass(frozen=True)
class StockPoolLocation:
    bucket: str | None
    key: str | None
    endpoint_url: str | None
    local_path: Path | None

    @property
    def is_remote(self) -> bool:
        return self.bucket is not None


@dataclass(frozen=True)
class StockPoolConfig:
    enabled: bool = False
    path: str = ""
    name: str = ""
    date_lag_sessions: int = 0
    filter_train: bool = False
    filter_selection: bool = False
    add_feature: bool = False
    annotate_predictions: bool = True
    membership_col: str = "stock_pool_member"


def stock_pool_config_from_mapping(config: dict) -> StockPoolConfig:
    path = config_str(config, "stock_pool", "path", "")
    configured_name = config_str(config, "stock_pool", "name", "")
    name = configured_name or (Path(path).stem if path else "")
    return StockPoolConfig(
        enabled=config_bool(config, "stock_pool", "enabled", False),
        path=path,
        name=name,
        date_lag_sessions=config_int(config, "stock_pool", "date_lag_sessions", 0),
        filter_train=config_bool(config, "stock_pool", "filter_train", False),
        filter_selection=config_bool(config, "stock_pool", "filter_selection", False),
        add_feature=config_bool(config, "stock_pool", "add_feature", False),
        annotate_predictions=config_bool(config, "stock_pool", "annotate_predictions", True),
        membership_col=config_str(
            config,
            "stock_pool",
            "membership_col",
            "stock_pool_member",
        ),
    )


def apply_stock_pool_cli_overrides(config: dict, args: Any) -> dict:
    pool = getattr(args, "pool", None)
    pool_path = getattr(args, "pool_path", None)
    pool_lag = getattr(args, "pool_date_lag_sessions", None)
    pool_filter_train = bool(getattr(args, "pool_filter_train", False))
    pool_add_feature = bool(getattr(args, "pool_add_feature", False))
    if (
        pool is None
        and pool_path in (None, "")
        and pool_lag is None
        and not pool_filter_train
        and not pool_add_feature
    ):
        return config

    out = {
        section: dict(values) if isinstance(values, dict) else values
        for section, values in config.items()
    }
    stock_pool = dict(out.get("stock_pool", {}))
    if pool:
        pool_name = str(pool).upper()
        stock_pool["enabled"] = True
        stock_pool["path"] = DEFAULT_STOCK_POOL_PATHS[pool_name]
        stock_pool["name"] = f"pool_{pool_name}"
    if pool_path not in (None, ""):
        stock_pool["enabled"] = True
        stock_pool["path"] = str(pool_path)
        stock_pool.setdefault("name", Path(str(pool_path)).stem)
    if pool or pool_path not in (None, ""):
        stock_pool.setdefault("filter_train", False)
        stock_pool.setdefault("filter_selection", True)
        stock_pool.setdefault("annotate_predictions", True)
        stock_pool.setdefault("membership_col", "stock_pool_member")
    if pool_lag is not None:
        stock_pool["date_lag_sessions"] = int(pool_lag)
    if pool_filter_train:
        stock_pool["filter_train"] = True
    if pool_add_feature:
        stock_pool["add_feature"] = True
    out["stock_pool"] = stock_pool
    return out


def parse_stock_pool_location(value: str | Path) -> StockPoolLocation:
    raw = str(value).strip()
    if not raw:
        raise SystemExit("stock pool path is empty")
    if "/" in raw and "@" in raw.split("/", 1)[0]:
        head, key = raw.split("/", 1)
        bucket, ceph_key = head.rsplit("@", 1)
        endpoint_url = CEPH_ENDPOINTS.get(ceph_key)
        if endpoint_url is None:
            valid = ", ".join(sorted(CEPH_ENDPOINTS))
            raise SystemExit(
                f"unknown stock pool Ceph endpoint {ceph_key!r}; expected one of: {valid}"
            )
        return StockPoolLocation(
            bucket=bucket,
            key=key,
            endpoint_url=endpoint_url,
            local_path=None,
        )
    return StockPoolLocation(bucket=None, key=None, endpoint_url=None, local_path=Path(raw))


def _build_ceph_access_key_id() -> str:
    user = os.environ.get("CEPH_LDAP_ID")
    password = os.environ.get("CEPH_LDAP_KEY")
    if not user or not password:
        raise SystemExit(
            "Missing Ceph S3 credentials. Set CEPH_LDAP_ID and CEPH_LDAP_KEY in .env "
            "and load it before reading lml.bzw@ssd stock pools."
        )
    token = {
        "RGW_TOKEN": {
            "version": 1,
            "type": "ldap",
            "id": user,
            "key": password,
        }
    }
    return base64.b64encode(json.dumps(token).encode("utf-8")).decode("utf-8")


def _read_remote_pool(location: StockPoolLocation) -> pd.DataFrame:
    try:
        import boto3
        import botocore
    except ImportError as exc:
        raise SystemExit(
            "Reading stock pools from Ceph S3 requires boto3. Install project "
            "dependencies or copy the pool parquet locally and use a local path."
        ) from exc
    client = boto3.client(
        service_name="s3",
        endpoint_url=location.endpoint_url,
        aws_access_key_id=_build_ceph_access_key_id(),
        aws_secret_access_key="",
        config=botocore.client.Config(s3={"addressing_style": "path"}),
    )
    response = client.get_object(Bucket=location.bucket, Key=location.key)
    return pd.read_parquet(BytesIO(response["Body"].read()))


def normalize_stock_pool_frame(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        raise SystemExit("stock pool frame is empty")
    out = pool.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    if out.index.nlevels != 1:
        raise SystemExit("stock pool frame must have a single date index")

    dates = pd.to_datetime(pd.Index(out.index), errors="coerce")
    if dates.isna().any():
        raise SystemExit("stock pool index contains non-date values")
    out.index = pd.Index(dates.strftime("%Y-%m-%d"), name="date")
    if out.index.has_duplicates:
        raise SystemExit("stock pool index contains duplicate dates")

    columns = normalize_symbols(pd.Series(out.columns, dtype="object"))
    if columns.duplicated().any():
        duplicated = sorted(set(columns.loc[columns.duplicated()].tolist()))
        raise SystemExit(f"stock pool columns duplicate after normalization: {duplicated[:5]}")
    out.columns = columns.tolist()
    return out.fillna(False).astype(bool).sort_index()


def load_stock_pool(path: str | Path) -> pd.DataFrame:
    location = parse_stock_pool_location(path)
    if location.is_remote:
        return normalize_stock_pool_frame(_read_remote_pool(location))
    if location.local_path is None or not location.local_path.exists():
        raise SystemExit(f"stock pool file does not exist: {location.local_path}")
    return normalize_stock_pool_frame(pd.read_parquet(location.local_path))


def load_configured_stock_pool(settings: StockPoolConfig) -> pd.DataFrame | None:
    if not settings.enabled:
        return None
    if not settings.path:
        raise SystemExit("[stock_pool].enabled=true requires [stock_pool].path")
    return load_stock_pool(settings.path)


def _pool_row_codes(
    dates: pd.Series,
    pool_index: pd.Index,
    *,
    date_lag_sessions: int = 0,
) -> np.ndarray:
    if date_lag_sessions < 0:
        raise SystemExit("stock_pool.date_lag_sessions must be >= 0")
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    if date_lag_sessions == 0:
        keys = parsed_dates.dt.strftime("%Y-%m-%d")
        return pool_index.get_indexer(keys)

    pool_dates = pd.DatetimeIndex(pd.to_datetime(pool_index))
    positions = pool_dates.searchsorted(parsed_dates, side="right") - 1
    positions = positions - int(date_lag_sessions)
    valid = parsed_dates.notna().to_numpy() & (positions >= 0)
    return np.where(valid, positions, -1).astype(np.int64)


def stock_pool_membership_mask(
    frame: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    date_lag_sessions: int = 0,
) -> pd.Series:
    if date_col not in frame.columns:
        raise SystemExit(f"stock pool filter missing required date column: {date_col}")
    if symbol_col not in frame.columns:
        raise SystemExit(f"stock pool filter missing required symbol column: {symbol_col}")
    normalized_pool = normalize_stock_pool_frame(pool)
    date_codes = _pool_row_codes(
        frame[date_col],
        normalized_pool.index,
        date_lag_sessions=date_lag_sessions,
    )
    symbols = normalize_symbols(frame[symbol_col])
    symbol_codes = pd.Index(normalized_pool.columns).get_indexer(symbols)
    valid = (date_codes >= 0) & (symbol_codes >= 0)

    mask = np.zeros(len(frame), dtype=bool)
    if valid.any():
        values = normalized_pool.to_numpy(dtype=bool, copy=False)
        mask[valid] = values[date_codes[valid], symbol_codes[valid]]
    return pd.Series(mask, index=frame.index, name="stock_pool_member")


def add_stock_pool_membership(
    frame: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    output_col: str = "stock_pool_member",
    date_lag_sessions: int = 0,
) -> pd.DataFrame:
    out = frame.copy()
    out[output_col] = stock_pool_membership_mask(
        out,
        pool,
        date_lag_sessions=date_lag_sessions,
    ).astype("int8")
    return out


def add_configured_stock_pool_feature(
    frame: pd.DataFrame,
    settings: StockPoolConfig,
    pool: pd.DataFrame | None,
) -> pd.DataFrame:
    if pool is None or not settings.add_feature:
        return frame
    return add_stock_pool_membership(
        frame,
        pool,
        output_col=settings.membership_col,
        date_lag_sessions=settings.date_lag_sessions,
    )


def filter_stock_pool_members(
    frame: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    date_lag_sessions: int = 0,
) -> pd.DataFrame:
    mask = stock_pool_membership_mask(
        frame,
        pool,
        date_lag_sessions=date_lag_sessions,
    )
    return frame.loc[mask].copy()


def filter_configured_stock_pool_train(
    train: pd.DataFrame,
    settings: StockPoolConfig,
    pool: pd.DataFrame | None,
) -> pd.DataFrame:
    if pool is None or not settings.filter_train:
        return train
    out = filter_stock_pool_members(
        train,
        pool,
        date_lag_sessions=settings.date_lag_sessions,
    )
    if out.empty:
        raise SystemExit("empty train frame after stock_pool.filter_train")
    return out


def stock_pool_membership_summary(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    prefix: str = "stock_pool",
) -> dict[str, object]:
    selected = frame.loc[mask]
    return {
        f"{prefix}_candidate_rows": int(len(selected)),
        f"{prefix}_candidate_row_fraction": (
            float(len(selected) / len(frame)) if len(frame) else float("nan")
        ),
        f"{prefix}_candidate_dates": (
            int(selected["date"].nunique()) if "date" in selected.columns else 0
        ),
        f"{prefix}_candidate_symbols": (
            int(selected["symbol"].nunique()) if "symbol" in selected.columns else 0
        ),
    }


def configured_stock_pool_selection_frame(
    predictions: pd.DataFrame,
    settings: StockPoolConfig,
    pool: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if pool is None:
        return predictions, predictions, {}

    mask = stock_pool_membership_mask(
        predictions,
        pool,
        date_lag_sessions=settings.date_lag_sessions,
    )
    out = predictions
    if settings.annotate_predictions or settings.filter_selection:
        out = predictions.copy()
        out[settings.membership_col] = mask.astype("int8").to_numpy()
        mask = out[settings.membership_col].astype(bool)

    summary = stock_pool_membership_summary(out, mask, prefix="stock_pool")
    summary.update(
        {
            "stock_pool_name": settings.name,
            "stock_pool_date_lag_sessions": settings.date_lag_sessions,
            "stock_pool_filter_selection": settings.filter_selection,
        }
    )
    if settings.filter_selection:
        return out, out.loc[mask].copy(), summary
    return out, out, summary


def stock_pool_runtime_summary(
    settings: StockPoolConfig,
    pool: pd.DataFrame,
) -> dict[str, object]:
    member_counts = pool.to_numpy(dtype=bool, copy=False).sum(axis=1)
    return {
        "enabled": True,
        "name": settings.name,
        "path": settings.path,
        "date_lag_sessions": settings.date_lag_sessions,
        "filter_train": settings.filter_train,
        "filter_selection": settings.filter_selection,
        "add_feature": settings.add_feature,
        "annotate_predictions": settings.annotate_predictions,
        "dates": int(len(pool.index)),
        "symbols": int(len(pool.columns)),
        "date_min": str(pool.index.min()),
        "date_max": str(pool.index.max()),
        "members_per_day_min": int(member_counts.min()) if len(member_counts) else 0,
        "members_per_day_median": (
            float(np.median(member_counts)) if len(member_counts) else float("nan")
        ),
        "members_per_day_max": int(member_counts.max()) if len(member_counts) else 0,
    }


def stock_pool_evaluation_settings(settings: StockPoolConfig) -> dict[str, object]:
    return {
        "stock_pool_enabled": settings.enabled,
        "stock_pool_name": settings.name,
        "stock_pool_path": settings.path,
        "stock_pool_date_lag_sessions": settings.date_lag_sessions,
        "stock_pool_filter_train": settings.filter_train,
        "stock_pool_filter_selection": settings.filter_selection,
        "stock_pool_add_feature": settings.add_feature,
        "stock_pool_annotate_predictions": settings.annotate_predictions,
    }
