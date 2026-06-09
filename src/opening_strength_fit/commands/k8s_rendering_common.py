from __future__ import annotations

import hashlib
import re
import textwrap

from opening_strength_fit.config import slug
from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT


def node_selector_yaml(config: dict, indent: int) -> str:
    node_selector = config.get("k8s", {}).get("node_selector", {})
    if not node_selector:
        return ""
    lines = [f"{' ' * indent}nodeSelector:"]
    for key, value in sorted(node_selector.items()):
        lines.append(f'{" " * (indent + 2)}{key}: "{value}"')
    return "\n".join(lines) + "\n"


def wait_for_specific_paths_yaml(
    paths: list[str],
    *,
    timeout_seconds: int,
    interval_seconds: int,
    indent: int,
) -> str:
    quoted_paths = " ".join(f'"{path.replace(chr(34), chr(92) + chr(34))}"' for path in paths)
    return textwrap.indent(
        textwrap.dedent(
            f"""\
            WAIT_PATH_TIMEOUT_SECONDS={timeout_seconds}
            WAIT_PATH_INTERVAL_SECONDS={interval_seconds}
            WAIT_PATH_STARTED=${{SECONDS}}
            WAIT_PATHS=({quoted_paths})
            for WAIT_PATH in "${{WAIT_PATHS[@]}}"; do
              until [ -f "${{WAIT_PATH}}" ]; do
                if [ $((SECONDS - WAIT_PATH_STARTED)) -ge "${{WAIT_PATH_TIMEOUT_SECONDS}}" ]; then
                  echo "timed out waiting for dependency file: ${{WAIT_PATH}}" >&2
                  exit 1
                fi
                echo "waiting for dependency file: ${{WAIT_PATH}}"
                sleep "${{WAIT_PATH_INTERVAL_SECONDS}}"
              done
              echo "dependency file is ready: ${{WAIT_PATH}}"
            done
            """
        ),
        " " * indent,
    )


def env_from_secrets_yaml(secret_names: list[str], indent: int) -> str:
    unique_names = []
    seen = set()
    for name in secret_names:
        item = str(name).strip()
        if item and item not in seen:
            unique_names.append(item)
            seen.add(item)
    if not unique_names:
        return ""
    lines = ["envFrom:"]
    for name in unique_names:
        lines.extend(["  - secretRef:", f"      name: {name}"])
    return textwrap.indent("\n".join(lines) + "\n", " " * indent)


def compact_run_slug(run_id_value: str, *, max_length: int) -> str:
    run_slug = slug(run_id_value)
    if len(run_slug) <= max_length:
        return run_slug

    tokens = [token for token in run_slug.split("-") if token]
    model = tokens[0] if tokens else ""
    delay = next((token for token in tokens if re.fullmatch(r"delay\d+", token)), "")
    horizon = next((token for token in tokens if re.fullmatch(r"\d+[mhd]", token)), "")
    weight = next((token for token in tokens if re.fullmatch(r"w\d+", token)), "")
    version = next((token for token in reversed(tokens) if re.fullmatch(r"v\d+", token)), "")

    candidates: list[list[str]] = []
    if "mixed" in tokens and weight:
        candidates.append([model, delay, horizon, "mixed", weight, version])
        candidates.append([model, delay, "mixed", weight])
        candidates.append(["mixed", weight, version])
    if "rolling" in tokens:
        candidates.append([model, delay, horizon, "roll", version])
        candidates.append([model, delay, "roll"])

    important = []
    for token in tokens:
        mapped = "roll" if token == "rolling" else token
        if (
            token == model
            or token in {"mixed", "top100"}
            or re.fullmatch(r"delay\d+", token)
            or re.fullmatch(r"\d+[mhd]", token)
            or re.fullmatch(r"w\d+", token)
            or re.fullmatch(r"v\d+", token)
            or token == "rolling"
        ):
            important.append(mapped)
    candidates.append(important)

    for parts in candidates:
        compact = "-".join(part for part in parts if part)
        if compact and len(compact) <= max_length:
            return compact

    return run_slug[:max_length].rstrip("-")


def k8s_job_name(
    prefix: str,
    run_id_value: str,
    suffix: str = "",
    *,
    max_length: int = KUBERNETES_NAME_LIMIT,
) -> str:
    parts = [prefix, slug(run_id_value)]
    if suffix:
        parts.append(suffix)
    candidate = "-".join(part.strip("-") for part in parts if part)
    if len(candidate) <= max_length:
        return candidate

    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    tail = f"-{suffix}-{digest}" if suffix else f"-{digest}"
    head = f"{prefix.strip('-')}-"
    keep = max_length - len(head) - len(tail)
    if keep < 1:
        keep = max_length - len(tail) - 1
        head = ""
    return f"{head}{compact_run_slug(run_id_value, max_length=keep)}{tail}"
