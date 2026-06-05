"""【功能描述】ReflACT 配置加载引擎：支持 `_base_` 继承的结构化 YAML 与扁平 legacy 格式。
【输入】YAML 配置文件路径；可选 `overrides`（`key=value` CLI 覆盖）。
【输出】合并后的配置 dict；`flatten_config(cfg)` 供 trainer 使用的扁平 dict。

支持两种配置格式：
  1. **结构化**（新）：`model`、`train`、`gradient`、`optimizer`、`evaluation`、`env` 等节，支持 ``_base_`` 继承。
  2. **扁平**（legacy）：所有键在顶层 — 完全向后兼容。

用法::

    from skillopt.config import load_config, flatten_config

    cfg = load_config("configs/searchqa_default.yaml")
    flat = flatten_config(cfg)  # 始终返回 trainer 所需的扁平 dict
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

# ── 标识结构化配置的节名 ──────────────────────────────────────────────────

_STRUCTURED_SECTIONS = frozenset({
    "model", "train", "gradient", "optimizer", "evaluation", "env",
})

# ── 结构化 → 扁平键映射 ────────────────────────────────────────────────────

_FLATTEN_MAP: dict[str, str] = {
    "model.backend": "model_backend",
    "model.optimizer": "optimizer_model",
    "model.target": "target_model",
    "model.optimizer_backend": "optimizer_backend",
    "model.target_backend": "target_backend",
    "model.reasoning_effort": "reasoning_effort",
    "model.rewrite_reasoning_effort": "rewrite_reasoning_effort",
    "model.rewrite_max_completion_tokens": "rewrite_max_completion_tokens",
    "model.codex_exec_path": "codex_exec_path",
    "model.codex_exec_sandbox": "codex_exec_sandbox",
    "model.codex_exec_profile": "codex_exec_profile",
    "model.codex_exec_full_auto": "codex_exec_full_auto",
    "model.codex_exec_reasoning_effort": "codex_exec_reasoning_effort",
    "model.codex_exec_use_sdk": "codex_exec_use_sdk",
    "model.codex_exec_network_access": "codex_exec_network_access",
    "model.codex_exec_web_search": "codex_exec_web_search",
    "model.codex_exec_approval_policy": "codex_exec_approval_policy",
    "model.claude_code_exec_path": "claude_code_exec_path",
    "model.claude_code_exec_profile": "claude_code_exec_profile",
    "model.claude_code_exec_use_sdk": "claude_code_exec_use_sdk",
    "model.claude_code_exec_effort": "claude_code_exec_effort",
    "model.claude_code_exec_max_thinking_tokens": "claude_code_exec_max_thinking_tokens",
    "model.codex_trace_to_optimizer": "codex_trace_to_optimizer",
    "model.azure_endpoint": "azure_endpoint",
    "model.azure_api_version": "azure_api_version",
    "model.azure_api_key": "azure_api_key",
    "model.azure_openai_endpoint": "azure_openai_endpoint",
    "model.azure_openai_api_version": "azure_openai_api_version",
    "model.azure_openai_api_key": "azure_openai_api_key",
    "model.azure_openai_auth_mode": "azure_openai_auth_mode",
    "model.azure_openai_ad_scope": "azure_openai_ad_scope",
    "model.azure_openai_managed_identity_client_id": "azure_openai_managed_identity_client_id",
    "model.optimizer_azure_openai_endpoint": "optimizer_azure_openai_endpoint",
    "model.optimizer_azure_openai_api_version": "optimizer_azure_openai_api_version",
    "model.optimizer_azure_openai_api_key": "optimizer_azure_openai_api_key",
    "model.optimizer_azure_openai_auth_mode": "optimizer_azure_openai_auth_mode",
    "model.optimizer_azure_openai_ad_scope": "optimizer_azure_openai_ad_scope",
    "model.optimizer_azure_openai_managed_identity_client_id": "optimizer_azure_openai_managed_identity_client_id",
    "model.target_azure_openai_endpoint": "target_azure_openai_endpoint",
    "model.target_azure_openai_api_version": "target_azure_openai_api_version",
    "model.target_azure_openai_api_key": "target_azure_openai_api_key",
    "model.target_azure_openai_auth_mode": "target_azure_openai_auth_mode",
    "model.target_azure_openai_ad_scope": "target_azure_openai_ad_scope",
    "model.target_azure_openai_managed_identity_client_id": "target_azure_openai_managed_identity_client_id",
    "model.qwen_chat_base_url": "qwen_chat_base_url",
    "model.qwen_chat_api_key": "qwen_chat_api_key",
    "model.qwen_chat_temperature": "qwen_chat_temperature",
    "model.qwen_chat_timeout_seconds": "qwen_chat_timeout_seconds",
    "model.qwen_chat_max_tokens": "qwen_chat_max_tokens",
    "model.qwen_chat_enable_thinking": "qwen_chat_enable_thinking",
    "train.num_epochs": "num_epochs",
    "train.train_size": "train_size",
    "train.steps_per_epoch": "steps_per_epoch",
    "train.batch_size": "batch_size",
    "train.accumulation": "accumulation",
    "train.seed": "seed",
    "gradient.minibatch_size": "minibatch_size",
    "gradient.merge_batch_size": "merge_batch_size",
    "gradient.analyst_workers": "analyst_workers",
    "gradient.failure_only": "failure_only",
    "gradient.max_analyst_rounds": "max_analyst_rounds",
    "optimizer.learning_rate": "edit_budget",
    "optimizer.min_learning_rate": "min_edit_budget",
    "optimizer.lr_scheduler": "lr_scheduler",
    "optimizer.lr_control_mode": "lr_control_mode",
    "optimizer.skill_update_mode": "skill_update_mode",
    "optimizer.meta_learning_rate": "meta_edit_budget",
    "optimizer.use_slow_update": "use_slow_update",
    "optimizer.slow_update_samples": "slow_update_samples",
    "optimizer.longitudinal_pair_policy": "longitudinal_pair_policy",
    "optimizer.use_meta_skill": "use_meta_skill",
    "evaluation.use_gate": "use_gate",
    "evaluation.sel_env_num": "sel_env_num",
    "evaluation.test_env_num": "test_env_num",
    "evaluation.eval_test": "eval_test",
    "env.name": "env",
    "env.skill_init": "skill_init",
    "env.out_root": "out_root",
}


# ── 深度合并 ───────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """将 *override* 递归合并进 *base*（返回新 dict）。"""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ── 带 _base_ 继承的 YAML 加载 ─────────────────────────────────────────

def _load_yaml(path: str, _visited: set[str] | None = None) -> dict:
    """加载 YAML 文件，递归解析 ``_base_`` 继承。"""
    abs_path = os.path.abspath(path)
    if _visited is None:
        _visited = set()
    if abs_path in _visited:
        raise ValueError(f"Circular _base_ inheritance: {abs_path}")
    _visited.add(abs_path)

    with open(abs_path) as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = os.path.join(os.path.dirname(abs_path), base_ref)
        base_cfg = _load_yaml(base_path, _visited)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


# ── 格式检测 ─────────────────────────────────────────────────────────────

def is_structured(cfg: dict) -> bool:
    """若 *cfg* 使用新的结构化节格式则返回 True。"""
    return any(
        key in _STRUCTURED_SECTIONS and isinstance(cfg.get(key), dict)
        for key in cfg
    )


# ── 扁平化 ──────────────────────────────────────────────────────────────

def flatten_config(cfg: dict) -> dict:
    """将结构化配置转为 trainer 期望的扁平 dict。

    若 *cfg* 已是扁平格式，则返回其浅拷贝且内容不变。
    """
    if not is_structured(cfg):
        return dict(cfg)

    flat: dict[str, Any] = {}

    evaluation_section = cfg.get("evaluation", {})
    if isinstance(evaluation_section, dict) and evaluation_section.get("use_gate") is False:
        raise ValueError(
            "Gate validation is mandatory in this branch. Remove "
            "`evaluation.use_gate: false` from the config."
        )

    # 应用显式映射
    for dotted, flat_key in _FLATTEN_MAP.items():
        section, key = dotted.split(".", 1)
        section_dict = cfg.get(section, {})
        if isinstance(section_dict, dict) and key in section_dict:
            flat[flat_key] = section_dict[key]

    # 透传未列入显式映射的环境专有键
    env_section = cfg.get("env", {})
    if isinstance(env_section, dict):
        mapped_env_keys = {
            k.split(".", 1)[1]
            for k in _FLATTEN_MAP
            if k.startswith("env.")
        }
        for key, val in env_section.items():
            if key not in mapped_env_keys:
                flat[key] = val

    return flat


# ── 覆盖项应用 ───────────────────────────────────────────────────────────

def _cast_value(val_str: str) -> Any:
    """将 CLI 字符串自动转换为 int / float / bool / str。"""
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """将 ``key=value`` 覆盖就地写入结构化配置。

    支持 ``section.key=value``（结构化配置）与
    ``key=value``（扁平配置或 env 节中的扁平键）。
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        key, val_str = item.split("=", 1)
        val = _cast_value(val_str)

        if "." in key:
            section, subkey = key.split(".", 1)
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section][subkey] = val
            else:
                cfg.setdefault(section, {})[subkey] = val
        else:
            # 扁平键 — 写入顶层（legacy 兼容）
            cfg[key] = val


# ── 公开 API ─────────────────────────────────────────────────────────────

def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> dict:
    """加载配置文件，支持 ``_base_`` 继承与可选覆盖项。

    Parameters
    ----------
    path : str
        YAML 配置文件路径。
    overrides : list[str] | None
        来自 ``--cfg-options`` 的 ``key=value`` 字符串列表。

    Returns
    -------
    dict
        合并后的配置（结构化或扁平，取决于 YAML 内容）。
    """
    cfg = _load_yaml(path)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg
