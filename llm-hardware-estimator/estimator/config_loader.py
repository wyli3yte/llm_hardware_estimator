import difflib
import json
from pathlib import Path


class ConfigError(Exception):
    pass


def load_mapping_file(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError("config file not found: %s" % path)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ConfigError(
                "%s is not JSON-compatible YAML and PyYAML is not installed" % path
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ConfigError("config must be a mapping: %s" % path)
    return payload


def project_root():
    return Path(__file__).resolve().parents[1]


def default_config_dir():
    return project_root() / "configs"


def load_project_configs(config_dir=None):
    base = Path(config_dir) if config_dir else default_config_dir()
    models = load_mapping_file(base / "models.yaml").get("models", {})
    hardware = load_mapping_file(base / "hardware.yaml").get("hardware", {})
    platforms = load_mapping_file(base / "platform_profiles.yaml").get("platform_profiles", {})
    scenarios = {}
    scenario_dir = base / "scenarios"
    if scenario_dir.exists():
        for path in sorted(scenario_dir.glob("*.yaml")):
            data = load_mapping_file(path)
            scenario = data.get("scenario", data)
            name = scenario.get("name", path.stem)
            scenarios[name] = scenario
    return {
        "models": models,
        "hardware": hardware,
        "platform_profiles": platforms,
        "scenarios": scenarios,
    }


def validate_project_configs(config_dir=None):
    configs = load_project_configs(config_dir)
    errors = []
    for model_id, model in configs["models"].items():
        for field in ("task_type", "params_billion"):
            if field not in model:
                errors.append("model %s missing %s" % (model_id, field))
    for hardware_id, gpu in configs["hardware"].items():
        for field in ("vendor", "model", "memory_gb", "precision_support"):
            if field not in gpu:
                errors.append("hardware %s missing %s" % (hardware_id, field))
    for scenario_id, scenario in configs["scenarios"].items():
        for field in ("input_tokens", "concurrency", "weight_precision"):
            if field not in scenario:
                errors.append("scenario %s missing %s" % (scenario_id, field))
    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "models": len(configs["models"]),
        "hardware": len(configs["hardware"]),
        "scenarios": len(configs["scenarios"]),
        "platform_profiles": len(configs["platform_profiles"]),
    }


def resolve_key(mapping, key, label):
    if key in mapping:
        return key, mapping[key]
    choices = difflib.get_close_matches(key, mapping.keys(), n=5, cutoff=0.35)
    suggestion = ""
    if choices:
        suggestion = "\n可能的近似匹配项：\n" + "\n".join(
            "%d. %s" % (idx, choice) for idx, choice in enumerate(choices, 1)
        )
    raise ConfigError("未找到%s：%s%s" % (label, key, suggestion))


def load_scenario(configs, scenario_or_path):
    candidate = Path(str(scenario_or_path))
    if candidate.exists():
        data = load_mapping_file(candidate)
        return data.get("scenario", data)
    return resolve_key(configs["scenarios"], scenario_or_path, "场景")[1]


def merge_scenario(scenario, overrides):
    merged = json.loads(json.dumps(scenario))
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
