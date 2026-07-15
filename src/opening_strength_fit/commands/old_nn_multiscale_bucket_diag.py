"""Compatibility exports for the archived multiscale diagnostic command."""

from opening_strength_fit.legacy.old_nn_multiscale_bucket_diag import (
    DEFAULT_MONTHS,
    DEFAULT_VARIANTS,
    main,
    parse_args,
    parse_csv_ints,
    parse_csv_strings,
)

__all__ = [
    "DEFAULT_MONTHS",
    "DEFAULT_VARIANTS",
    "main",
    "parse_args",
    "parse_csv_ints",
    "parse_csv_strings",
]
