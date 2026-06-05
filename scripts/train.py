#!/usr/bin/env python3
"""【功能描述】SkillOpt 统一训练入口，加载配置并启动 ReflACT 训练循环。
【输入】命令行参数（--config、--cfg-options 及遗留扁平覆盖项）、YAML 配置文件。
【输出】在 out_root 下写入训练产物与 summary；控制台打印运行摘要。

用法
-----
    python scripts/train.py --config configs/t2i/default.yaml

任意 YAML 键均可从命令行覆盖::

    python scripts/train.py --config configs/t2i/default.yaml \\
        --batch_size 40 --num_epochs 2 --seed 123

运行 ``python scripts/train.py --help`` 查看完整选项列表。
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

# 将项目根目录加入 sys.path，以便无论从何处调用脚本都能 ``import skillopt``
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.model.common import default_model_for_backend, normalize_backend_name

_OPENAI_DEFAULT_MODEL_SENTINELS = {"gpt-5.4", "gpt-5.5"}


from skillopt.envs.registry import get_adapter  # noqa: E402


# ── 命令行接口 ──────────────────────────────────────────────────────────────

_BOOL = lambda x: x.lower() in ("true", "1", "yes")  # noqa: E731


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SkillOpt: Executive Strategy for Self-Evolving Agent Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--cfg-options", nargs="+", default=[],
                   help="Override config: section.key=value (e.g. train.batch_size=40)")

    # 遗留扁平 CLI 覆盖（仍可用，新用法优先 --cfg-options）
    p.add_argument("--env", type=str)
    p.add_argument("--backend", type=str,
                   choices=["azure_openai", "codex", "codex_exec", "claude", "claude_chat", "claude_code_exec", "qwen", "qwen_chat"])
    p.add_argument("--optimizer_model", type=str)
    p.add_argument("--target_model", type=str)
    p.add_argument("--optimizer_backend", type=str)
    p.add_argument("--target_backend", type=str)
    p.add_argument("--reasoning_effort", type=str,
                   choices=["", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--rewrite_reasoning_effort", type=str)
    p.add_argument("--rewrite_max_completion_tokens", type=int)
    p.add_argument("--azure_endpoint", type=str)
    p.add_argument("--azure_api_version", type=str)
    p.add_argument("--azure_api_key", type=str)
    p.add_argument("--azure_openai_endpoint", type=str)
    p.add_argument("--azure_openai_api_version", type=str)
    p.add_argument("--azure_openai_api_key", type=str)
    p.add_argument("--azure_openai_auth_mode", type=str)
    p.add_argument("--azure_openai_ad_scope", type=str)
    p.add_argument("--azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--optimizer_azure_openai_endpoint", type=str)
    p.add_argument("--optimizer_azure_openai_api_version", type=str)
    p.add_argument("--optimizer_azure_openai_api_key", type=str)
    p.add_argument("--optimizer_azure_openai_auth_mode", type=str)
    p.add_argument("--optimizer_azure_openai_ad_scope", type=str)
    p.add_argument("--optimizer_azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--target_azure_openai_endpoint", type=str)
    p.add_argument("--target_azure_openai_api_version", type=str)
    p.add_argument("--target_azure_openai_api_key", type=str)
    p.add_argument("--target_azure_openai_auth_mode", type=str)
    p.add_argument("--target_azure_openai_ad_scope", type=str)
    p.add_argument("--target_azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--qwen_chat_base_url", type=str)
    p.add_argument("--qwen_chat_api_key", type=str)
    p.add_argument("--qwen_chat_temperature", type=float)
    p.add_argument("--qwen_chat_timeout_seconds", type=float)
    p.add_argument("--qwen_chat_max_tokens", type=int)
    p.add_argument("--qwen_chat_enable_thinking", type=_BOOL)
    p.add_argument("--codex_exec_path", type=str)
    p.add_argument("--codex_exec_sandbox", type=str)
    p.add_argument("--codex_exec_profile", type=str)
    p.add_argument("--codex_exec_full_auto", type=_BOOL)
    p.add_argument("--codex_exec_reasoning_effort", type=str)
    p.add_argument("--codex_exec_use_sdk", type=str)
    p.add_argument("--codex_exec_network_access", type=_BOOL)
    p.add_argument("--codex_exec_web_search", type=_BOOL)
    p.add_argument("--codex_exec_approval_policy", type=str)
    p.add_argument("--claude_code_exec_path", type=str)
    p.add_argument("--claude_code_exec_profile", type=str)
    p.add_argument("--claude_code_exec_use_sdk", type=str)
    p.add_argument("--claude_code_exec_effort", type=str)
    p.add_argument("--claude_code_exec_max_thinking_tokens", type=int)
    p.add_argument("--codex_trace_to_optimizer", type=_BOOL)
    p.add_argument("--prompt_init", type=str)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--train_size", type=int)
    p.add_argument("--steps_per_epoch", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--edit_budget", type=int)
    p.add_argument("--min_edit_budget", type=int)
    p.add_argument("--lr_scheduler", type=str,
                   choices=["constant", "linear", "cosine", "autonomous"])
    p.add_argument("--lr_control_mode", type=str,
                   choices=["fixed", "autonomous", "none"])
    p.add_argument("--merge_batch_size", type=int)
    p.add_argument("--max_analyst_rounds", type=int)
    p.add_argument("--sel_env_num", type=int)
    p.add_argument("--test_env_num", type=int)
    p.add_argument("--eval_test", type=_BOOL)
    p.add_argument("--use_gate", type=_BOOL)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--max_api_workers", type=int)
    p.add_argument("--analyst_workers", type=int)
    p.add_argument("--failure_only", type=_BOOL)
    p.add_argument("--minibatch_size", type=int)
    p.add_argument("--prompt_update_mode", type=str,
                   choices=[
                       "patch",
                       "rewrite_from_suggestions",
                       "rewrite",
                       "suggestions",
                       "full_rewrite",
                       "full_rewrite_minibatch",
                       "minibatch_full_rewrite",
                   ])
    p.add_argument("--use_slow_update", type=_BOOL)
    p.add_argument("--slow_update_samples", type=int)
    p.add_argument("--longitudinal_pair_policy", type=str,
                   choices=["mixed", "changed", "unchanged"])
    p.add_argument("--use_meta_prompt", type=_BOOL)
    p.add_argument("--data_path", type=str)
    p.add_argument("--split_mode", type=str,
                   choices=["ratio", "split_dir"])
    p.add_argument("--split_ratio", type=str)
    p.add_argument("--split_seed", type=int)
    p.add_argument("--split_dir", type=str)
    p.add_argument("--split_output_dir", type=str)
    p.add_argument("--data_root", type=str)
    p.add_argument("--max_turns", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--shuffle_choices", type=_BOOL)
    p.add_argument("--use_theorem", type=_BOOL)
    p.add_argument("--use_sketch", type=_BOOL)
    p.add_argument("--image_detail", type=str)
    p.add_argument("--judge_model", type=str)
    p.add_argument("--judge_max_completion_tokens", type=int)
    p.add_argument("--judge_retries", type=int)
    p.add_argument("--out_root", type=str)
    p.add_argument("--mode", type=str)

    return p.parse_args()


# ── 扁平键 → 结构化路径映射（遗留 CLI → 结构化配置）────────────────────────

_LEGACY_TO_STRUCTURED: dict[str, str] = {
    "backend": "model.backend",
    "optimizer_model": "model.optimizer",
    "target_model": "model.target",
    "optimizer_backend": "model.optimizer_backend",
    "target_backend": "model.target_backend",
    "reasoning_effort": "model.reasoning_effort",
    "rewrite_reasoning_effort": "model.rewrite_reasoning_effort",
    "rewrite_max_completion_tokens": "model.rewrite_max_completion_tokens",
    "azure_endpoint": "model.azure_endpoint",
    "azure_api_version": "model.azure_api_version",
    "azure_api_key": "model.azure_api_key",
    "azure_openai_endpoint": "model.azure_openai_endpoint",
    "azure_openai_api_version": "model.azure_openai_api_version",
    "azure_openai_api_key": "model.azure_openai_api_key",
    "azure_openai_auth_mode": "model.azure_openai_auth_mode",
    "azure_openai_ad_scope": "model.azure_openai_ad_scope",
    "azure_openai_managed_identity_client_id": "model.azure_openai_managed_identity_client_id",
    "optimizer_azure_openai_endpoint": "model.optimizer_azure_openai_endpoint",
    "optimizer_azure_openai_api_version": "model.optimizer_azure_openai_api_version",
    "optimizer_azure_openai_api_key": "model.optimizer_azure_openai_api_key",
    "optimizer_azure_openai_auth_mode": "model.optimizer_azure_openai_auth_mode",
    "optimizer_azure_openai_ad_scope": "model.optimizer_azure_openai_ad_scope",
    "optimizer_azure_openai_managed_identity_client_id": "model.optimizer_azure_openai_managed_identity_client_id",
    "target_azure_openai_endpoint": "model.target_azure_openai_endpoint",
    "target_azure_openai_api_version": "model.target_azure_openai_api_version",
    "target_azure_openai_api_key": "model.target_azure_openai_api_key",
    "target_azure_openai_auth_mode": "model.target_azure_openai_auth_mode",
    "target_azure_openai_ad_scope": "model.target_azure_openai_ad_scope",
    "target_azure_openai_managed_identity_client_id": "model.target_azure_openai_managed_identity_client_id",
    "qwen_chat_base_url": "model.qwen_chat_base_url",
    "qwen_chat_api_key": "model.qwen_chat_api_key",
    "qwen_chat_temperature": "model.qwen_chat_temperature",
    "qwen_chat_timeout_seconds": "model.qwen_chat_timeout_seconds",
    "qwen_chat_max_tokens": "model.qwen_chat_max_tokens",
    "qwen_chat_enable_thinking": "model.qwen_chat_enable_thinking",
    "codex_exec_path": "model.codex_exec_path",
    "codex_exec_sandbox": "model.codex_exec_sandbox",
    "codex_exec_profile": "model.codex_exec_profile",
    "codex_exec_full_auto": "model.codex_exec_full_auto",
    "codex_exec_reasoning_effort": "model.codex_exec_reasoning_effort",
    "codex_exec_use_sdk": "model.codex_exec_use_sdk",
    "codex_exec_network_access": "model.codex_exec_network_access",
    "codex_exec_web_search": "model.codex_exec_web_search",
    "codex_exec_approval_policy": "model.codex_exec_approval_policy",
    "claude_code_exec_path": "model.claude_code_exec_path",
    "claude_code_exec_profile": "model.claude_code_exec_profile",
    "claude_code_exec_use_sdk": "model.claude_code_exec_use_sdk",
    "claude_code_exec_effort": "model.claude_code_exec_effort",
    "claude_code_exec_max_thinking_tokens": "model.claude_code_exec_max_thinking_tokens",
    "codex_trace_to_optimizer": "model.codex_trace_to_optimizer",
    "num_epochs": "train.num_epochs",
    "train_size": "train.train_size",
    "steps_per_epoch": "train.steps_per_epoch",
    "batch_size": "train.batch_size",
    "accumulation": "train.accumulation",
    "seed": "train.seed",
    "minibatch_size": "gradient.minibatch_size",
    "merge_batch_size": "gradient.merge_batch_size",
    "analyst_workers": "gradient.analyst_workers",
    "max_analyst_rounds": "gradient.max_analyst_rounds",
    "failure_only": "gradient.failure_only",
    "edit_budget": "optimizer.learning_rate",
    "min_edit_budget": "optimizer.min_learning_rate",
    "lr_scheduler": "optimizer.lr_scheduler",
    "lr_control_mode": "optimizer.lr_control_mode",
    "prompt_update_mode": "optimizer.prompt_update_mode",
    "use_slow_update": "optimizer.use_slow_update",
    "slow_update_samples": "optimizer.slow_update_samples",
    "longitudinal_pair_policy": "optimizer.longitudinal_pair_policy",
    "use_meta_prompt": "optimizer.use_meta_prompt",
    "use_gate": "evaluation.use_gate",
    "sel_env_num": "evaluation.sel_env_num",
    "test_env_num": "evaluation.test_env_num",
    "eval_test": "evaluation.eval_test",
    "env": "env.name",
    "prompt_init": "env.prompt_init",
    "out_root": "env.out_root",
}


def load_config(args: argparse.Namespace) -> dict:
    """加载含 _base_ 继承的配置，再应用 CLI 覆盖。"""
    from skillopt.config import load_config as _load, flatten_config, is_structured

    cfg = _load(args.config, overrides=args.cfg_options)
    structured = is_structured(cfg)

    # 应用遗留 --key value 覆盖
    cli = {k: v for k, v in vars(args).items()
           if v is not None and k not in ("config", "cfg_options")}
    if cli:
        if structured:
            from skillopt.config import apply_overrides
            mapped = []
            for k, v in cli.items():
                dotted = _LEGACY_TO_STRUCTURED.get(k)
                if dotted:
                    mapped.append(f"{dotted}={v}")
                else:
                    mapped.append(f"env.{k}={v}")
            apply_overrides(cfg, mapped)
        else:
            cfg.update(cli)

    # 结构化配置展平为 flat dict，供 trainer/adapter 使用
    flat = flatten_config(cfg) if structured else cfg

    for new_key, old_key in (
        ("azure_openai_endpoint", "azure_endpoint"),
        ("azure_openai_api_version", "azure_api_version"),
        ("azure_openai_api_key", "azure_api_key"),
    ):
        if flat.get(new_key) in (None, "") and flat.get(old_key) not in (None, ""):
            flat[new_key] = flat[old_key]

    explicit_backend = getattr(args, "backend", None)
    if explicit_backend is None:
        for option in args.cfg_options or []:
            key = str(option).split("=", 1)[0].strip()
            if key == "model.backend":
                explicit_backend = str(option).split("=", 1)[1].strip()
                break

    backend = normalize_backend_name(flat.get("model_backend") or flat.get("target_backend") or "azure_openai")

    def _has_model_override(dotted_key: str, legacy_key: str) -> bool:
        if getattr(args, legacy_key, None) is not None:
            return True
        for option in args.cfg_options or []:
            key = str(option).split("=", 1)[0].strip()
            if key == dotted_key:
                return True
        return False

    if explicit_backend is not None:
        backend = normalize_backend_name(explicit_backend)
        flat["model_backend"] = backend
        if backend in {"claude", "claude_chat"}:
            flat.setdefault("optimizer_backend", "claude_chat")
            flat.setdefault("target_backend", "claude_chat")
        elif backend in {"codex", "codex_exec"}:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "codex_exec")
        elif backend == "claude_code_exec":
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "claude_code_exec")
        elif backend in {"qwen", "qwen_chat"}:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "qwen_chat")
        else:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "openai_chat")
    else:
        flat.setdefault("optimizer_backend", "openai_chat")
        flat.setdefault("target_backend", "openai_chat")

    if flat.get("optimizer_backend") == "claude_chat":
        if (
            str(flat.get("optimizer_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.optimizer", "optimizer_model")
        ):
            flat["optimizer_model"] = default_model_for_backend("claude_chat")
    if flat.get("target_backend") == "claude_chat":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = default_model_for_backend("claude_chat")
    if flat.get("target_backend") == "claude_code_exec":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = default_model_for_backend("claude_chat")
    if flat.get("target_backend") == "qwen_chat":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = default_model_for_backend("qwen_chat")

    # 自动生成输出根目录
    if not flat.get("out_root"):
        env = flat.get("env", "unknown")
        model = flat.get("optimizer_model", "unknown").replace("/", "-")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        flat["out_root"] = os.path.join("outputs", f"skillopt_{env}_{model}_{ts}")

    flat["out_root"] = os.path.abspath(flat["out_root"])
    return flat


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    print(f"\n{'='*60}")
    print(f"  SkillOpt — Executive Strategy for Self-Evolving Agent Skills")
    print(f"{'='*60}")
    print(f"  env:            {cfg.get('env')}")
    print(f"  optimizer_model:  {cfg.get('optimizer_model')}")
    print(f"  target_model:  {cfg.get('target_model')}")
    print(f"  optimizer_backend:{cfg.get('optimizer_backend', 'openai_chat')}")
    print(f"  target_backend:{cfg.get('target_backend', 'openai_chat')}")
    print(f"  reasoning:      {cfg.get('reasoning_effort') or 'off'}")
    print(f"  rewrite_effort: {cfg.get('rewrite_reasoning_effort') or 'off'}")
    print(f"  epochs:         {cfg.get('num_epochs')}")
    print(f"  train_size:     {cfg.get('train_size') or 'from dataset'}")
    print(f"  steps/epoch:    auto")
    print(f"  batch_size:     {cfg.get('batch_size')}")
    print(f"  edit_budget:    {cfg.get('edit_budget')}")
    print(f"  lr_scheduler:   {cfg.get('lr_scheduler', 'constant')}")
    print(f"  update_mode:    {cfg.get('prompt_update_mode', 'patch')}")
    print(f"  min_edit_budget:{cfg.get('min_edit_budget', 2)}")
    print(f"  minibatch_size: {cfg.get('minibatch_size')}")
    print(f"  seed:           {cfg.get('seed')}")
    print(f"  meta_prompt:    {cfg.get('use_meta_prompt', False)}")
    print(f"  slow_update:    {cfg.get('use_slow_update', False)}")
    print(f"  out_root:       {cfg.get('out_root')}")
    print(f"{'='*60}\n")

    # 构建 adapter
    adapter = get_adapter(cfg)

    # 构建 trainer 并运行
    from skillopt.engine.trainer import ReflACTTrainer
    trainer = ReflACTTrainer(cfg, adapter)
    summary = trainer.train()

    print(f"\n  Output saved to: {cfg['out_root']}")
    if summary.get("test_hard") is not None:
        print(f"  Final test: {summary['test_hard']:.4f}")


if __name__ == "__main__":
    main()
