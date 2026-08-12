from __future__ import annotations

from opening_strength_fit.commands.execution_context import (
    _output_root,
)
from opening_strength_fit.commands.execution_context import (
    ask_level_main as main,
)
from opening_strength_fit.commands.execution_context import (
    parse_ask_level_args as parse_args,
)

__all__ = ["_output_root", "main", "parse_args"]


if __name__ == "__main__":
    main()
