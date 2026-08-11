from __future__ import annotations

import json
from pathlib import Path

from opening_strength_fit.support_jobs import load_support_jobs, main, render_support_job

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/jobs/support/ds350_script_jobs.toml"


def test_support_job_spec_renders_all_jobs() -> None:
    defaults, jobs = load_support_jobs(SPEC)

    assert len(jobs) == 22
    assert len({job["name"] for job in jobs.values()}) == 22
    for job in jobs.values():
        manifest = render_support_job(defaults, job)
        assert manifest["kind"] == "Job"
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 172800
        command = manifest["spec"]["template"]["spec"]["containers"][0]["command"]
        if job.get("matrix"):
            assert command == ["/bin/bash", "-lc"]
        else:
            assert command[:2] == ["python", job["script"]]


def test_support_job_optional_gpu_secret_and_output() -> None:
    defaults, jobs = load_support_jobs(SPEC)
    gpu_pod = render_support_job(defaults, jobs["model-tradeability"])["spec"]["template"]["spec"]
    assert gpu_pod["nodeSelector"] == {"has_gpu": "true"}
    assert gpu_pod["containers"][0]["envFrom"] == [
        {"secretRef": {"name": "xy-fit-ceph-credentials"}}
    ]

    probe_pod = render_support_job(defaults, jobs["stock-pool-probe"])["spec"]["template"]["spec"]
    assert all(volume["name"] != "output" for volume in probe_pod["volumes"])
    assert all(mount["name"] != "output" for mount in probe_pod["containers"][0]["volumeMounts"])

    indexed = render_support_job(defaults, jobs["label-scale"])
    assert indexed["spec"]["completions"] == 10
    shell = indexed["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert 'WINDOW="${WINDOW_VALUES[${JOB_COMPLETION_INDEX:' in shell
    assert 'HORIZON="${HORIZON_VALUES[${JOB_COMPLETION_INDEX:' in shell
    assert '--window "${WINDOW}" --horizon "${HORIZON}"' in shell


def test_support_job_cli_writes_selected_manifest(tmp_path: Path) -> None:
    output = tmp_path / "job.json"
    assert main([str(SPEC), "--job", "overnight", "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["metadata"]["name"] == "os-ds350-2022-2025-all-a-overnight-v1"
