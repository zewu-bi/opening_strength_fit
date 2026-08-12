from __future__ import annotations

from opening_strength_fit import k8s_rendering_support as _impl

__all__ = """avoid_nodes_affinity_yaml compact_run_slug env_from_secrets_yaml k8s_job_name
node_selector_yaml training_config_map_mount_yaml training_config_map_volume_yaml
wait_for_specific_paths_yaml""".split()
globals().update({name: getattr(_impl, name) for name in __all__})
