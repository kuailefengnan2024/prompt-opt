"""【功能描述】ReflACT 训练器，主编排 6 阶段 ReflACT 流水线。
【输入】cfg 配置字典、adapter 环境适配器实例。
【输出】训练产物（prompts、history、summary 等）及 summary 字典。

流水线阶段：
  1. 推演（Rollout）— 用当前 prompt 执行 episode
  2. 反思（Reflect）— 分析轨迹并生成 patch
  3. 聚合（Aggregate）— 分层合并 patch
  4. 筛选（Select）— 排序并选取 top 编辑
  5. 更新（Update）— 将编辑应用到 prompt 文档
  6. 评估（Evaluate）— 验证候选 prompt，接受/拒绝

训练器与环境无关；环境相关逻辑均委托给
:class:`~skillopt.envs.base.EnvAdapter`。
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.evaluation.gate import evaluate_gate
from skillopt.gradient.aggregate import merge_patches
from skillopt.optimizer.meta_prompt import run_meta_prompt
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.rewrite import rewrite_prompt_from_suggestions
from skillopt.optimizer.scheduler import build_scheduler
from skillopt.optimizer.prompt_editor import apply_patch_with_report
from skillopt.optimizer.slow_update import (
    build_comparison_pairs,
    extract_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_label,
    short_item_summary,
)
from skillopt.model import (
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    configure_qwen_chat,
    get_token_summary,
    reset_token_tracker,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
    set_optimizer_backend,
    set_optimizer_deployment,
)
from skillopt.utils import compute_score, prompt_hash


# ── Prompt 产物路径（T2I 用 best_prompt.md；上游默认仍为 best_prompt.md）────────

@dataclass(frozen=True)
class PromptArtifacts:
    """训练产物文件名约定；由 env 配置覆盖。"""

    best_file: str = "best_prompt.md"
    version_dir: str = "prompts"
    version_prefix: str = "prompt_v"


def _artifact_layout(cfg: dict) -> PromptArtifacts:
    return PromptArtifacts(
        best_file=str(cfg.get("best_prompt_file") or "best_prompt.md"),
        version_dir=str(cfg.get("prompt_version_dir") or "prompts"),
        version_prefix=str(cfg.get("prompt_version_prefix") or "prompt_v"),
    )


def _checkpoint_path(out_root: str, step: int, art: PromptArtifacts) -> str:
    return os.path.join(
        out_root, art.version_dir, f"{art.version_prefix}{step:04d}.md",
    )


def _best_prompt_path(out_root: str, art: PromptArtifacts) -> str:
    primary = os.path.join(out_root, art.best_file)
    if os.path.exists(primary):
        return primary
    legacy = os.path.join(out_root, "best_skill.md")
    return legacy if os.path.exists(legacy) else primary


def _resolve_prompt_init_path(cfg: dict) -> str:
    path = cfg.get("prompt_init") or ""
    return os.path.abspath(path) if path else ""


# ── Patch 规范化 ─────────────────────────────────────────────────────────────

def _normalise_patches(
    raw_patches: list[dict | None],
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict]]:
    """提取内层 patch 子字典，拆分为 failure/success 列表。

    每个元素应符合 :class:`~skillopt.types.RawPatch`。
    """
    mode = normalize_update_mode(update_mode)
    failure: list[dict] = []
    success: list[dict] = []
    for p in raw_patches:
        if not isinstance(p, dict):
            continue
        inner = p.get("patch", p)
        if not isinstance(inner, dict):
            continue
        items = get_payload_items(inner, mode)
        if not items:
            continue
        support = max(int(p.get("batch_size", 0) or 0), 1)
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_type", p.get("source_type", "failure"))
                item.setdefault("support_count", support)
        if p.get("source_type", "failure") == "success":
            success.append(inner)
        else:
            failure.append(inner)
    return failure, success


def _normalise_longitudinal_pair_policy(policy: str | None) -> str:
    raw = str(policy or "mixed").strip().lower()
    aliases = {
        "mixed": "mixed",
        "default": "mixed",
        "random": "mixed",
        "all": "mixed",
        "changed": "changed",
        "change": "changed",
        "delta": "changed",
        "10_01": "changed",
        "01_10": "changed",
        "unchanged": "unchanged",
        "stable": "unchanged",
        "same": "unchanged",
        "00_11": "unchanged",
    }
    if raw not in aliases:
        raise ValueError(
            "optimizer.longitudinal_pair_policy must be one of "
            "mixed, changed, unchanged"
        )
    return aliases[raw]


def _normalise_lr_control_mode(mode: str | None) -> str:
    raw = str(mode or "fixed").strip().lower()
    aliases = {
        "fixed": "fixed",
        "manual": "fixed",
        "scheduler": "fixed",
        "scheduled": "fixed",
        "autonomous": "autonomous",
        "auto": "autonomous",
        "optimizer": "autonomous",
        "none": "none",
        "off": "none",
        "no_lr": "none",
    }
    if raw not in aliases:
        raise ValueError("optimizer.lr_control_mode must be one of fixed, autonomous, none")
    return aliases[raw]


def _filter_longitudinal_pairs(pairs: list[dict], policy: str) -> list[dict]:
    if policy == "mixed":
        return pairs
    if policy == "changed":
        keep = {"improved", "regressed"}
    elif policy == "unchanged":
        keep = {"persistent_fail", "stable_success"}
    else:
        raise ValueError(f"Unknown longitudinal pair policy: {policy}")
    return [p for p in pairs if p.get("category") in keep]


def _pair_category_counts(pairs: list[dict]) -> dict[str, int]:
    counts = {
        "improved": 0,
        "regressed": 0,
        "persistent_fail": 0,
        "stable_success": 0,
    }
    for pair in pairs:
        cat = str(pair.get("category", ""))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _safe_pair_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe[:80] or "item"


def _build_longitudinal_pairs(
    *,
    adapter: EnvAdapter,
    dataloader,
    prev_prompt: str,
    curr_prompt: str,
    initial_items: list[dict],
    initial_prev_results: list[dict],
    initial_curr_results: list[dict],
    prev_rollout_dir: str,
    curr_rollout_dir: str,
    policy: str,
    target_n: int,
    seed: int,
    out_root: str,
) -> tuple[list[dict], list[dict]]:
    """构建纵向对比对，可按变化类别过滤。

    ``mixed`` 完全保留旧行为。``changed`` 仅保留 10/01 对，并尝试扫描 train
    划分一次以补足至 ``target_n``。``unchanged`` 仅保留 00/11 对且不补足。
    """
    all_pairs = build_comparison_pairs(
        initial_prev_results,
        initial_curr_results,
        initial_items,
        prev_rollout_dir=prev_rollout_dir,
        curr_rollout_dir=curr_rollout_dir,
    )
    selected_pairs = _filter_longitudinal_pairs(all_pairs, policy)
    if policy != "changed" or len(selected_pairs) >= target_n or dataloader is None:
        return selected_pairs, all_pairs

    train_items = list(getattr(dataloader, "train_items", []) or [])
    if not train_items:
        return selected_pairs, all_pairs

    seen_ids = {str(p.get("id", "")) for p in all_pairs}
    rng = random.Random(seed)
    candidates = list(train_items)
    rng.shuffle(candidates)
    candidates = [item for item in candidates if str(item.get("id", "")) not in seen_ids]

    for idx, item in enumerate(candidates):
        if len(selected_pairs) >= target_n:
            break
        item_id = _safe_pair_id(str(item.get("id", f"item_{idx}")))
        batch = BatchSpec(
            phase="train",
            split="train",
            seed=seed + idx + 1,
            batch_size=1,
            payload=[item],
        )
        env = adapter.build_env_from_batch(batch, out_root=out_root)
        prev_dir = os.path.join(prev_rollout_dir, "topup", item_id)
        curr_dir = os.path.join(curr_rollout_dir, "topup", item_id)
        prev_results = adapter.rollout(env, prev_prompt, prev_dir)
        curr_results = adapter.rollout(env, curr_prompt, curr_dir)
        pair = build_comparison_pairs(
            prev_results,
            curr_results,
            [item],
            prev_rollout_dir=prev_dir,
            curr_rollout_dir=curr_dir,
        )
        all_pairs.extend(pair)
        selected_pairs.extend(_filter_longitudinal_pairs(pair, policy))

    return selected_pairs[:target_n], all_pairs


# ── 历史记录 / 持久化辅助 ─────────────────────────────────────────────────────

_SECRET_KEYS = {
    "azure_api_key",
    "api_key",
    "openai_api_key",
}


def _redact_value(val: str) -> str:
    if len(val) <= 8:
        return "*" * len(val)
    return f"{val[:4]}...{val[-4:]}"


def _redact_cfg(cfg: dict) -> dict:
    redacted = dict(cfg)
    for key in list(redacted):
        if key.lower() in _SECRET_KEYS and redacted.get(key):
            redacted[key] = _redact_value(str(redacted[key]))
    return redacted

def _load_history(out_root: str) -> list[dict]:
    path = os.path.join(out_root, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_history(out_root: str, history: list[dict]) -> None:
    path = os.path.join(out_root, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _save_prompt(
    out_root: str, step: int, content: str, art: PromptArtifacts | None = None,
) -> None:
    layout = art or PromptArtifacts()
    version_dir = os.path.join(out_root, layout.version_dir)
    os.makedirs(version_dir, exist_ok=True)
    with open(_checkpoint_path(out_root, step, layout), "w") as f:
        f.write(content)


def _load_prompt(out_root: str, step: int, art: PromptArtifacts | None = None) -> str:
    layout = art or PromptArtifacts()
    with open(_checkpoint_path(out_root, step, layout)) as f:
        return f.read()


def _load_meta_prompt_content(out_root: str, epoch: int) -> str:
    if epoch <= 0:
        return ""
    for dirname, fname, key in (
        ("meta_prompt", "meta_prompt_result.json", "meta_prompt_content"),
        ("meta_skill", "meta_skill_result.json", "meta_skill_content"),
    ):
        path = os.path.join(out_root, dirname, f"epoch_{epoch:02d}", fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                result = json.load(f)
            return str(result.get(key, "")).strip()
        except Exception:
            return ""
    return ""


def _load_runtime_state(out_root: str) -> dict | None:
    path = os.path.join(out_root, "runtime_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def _save_runtime_state(out_root: str, state: dict) -> None:
    path = os.path.join(out_root, "runtime_state.json")
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _resolve_train_size(cfg: dict, dataloader) -> int:
    configured = int(cfg.get("train_size", 0) or 0)
    inferred: int | None = None

    if dataloader is not None:
        getter = getattr(dataloader, "get_train_size", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if value is not None:
                inferred = int(value)
        elif hasattr(dataloader, "train_items"):
            try:
                inferred = len(getattr(dataloader, "train_items"))
            except Exception:
                inferred = None

    if inferred is not None and inferred <= 0:
        inferred = None

    if configured > 0 and inferred is not None and configured != inferred:
        raise ValueError(
            f"Configured train_size={configured} does not match loaded train split "
            f"size={inferred}. Fix the config or the dataset split."
        )

    train_size = configured if configured > 0 else inferred
    if train_size is None or train_size <= 0:
        raise ValueError(
            "Unable to determine train_size automatically. "
            "Provide train.train_size in the config for this environment."
        )
    return int(train_size)


def _compute_task_type_buckets(results: list[dict], task_types: list[str]) -> dict[str, dict]:
    """计算各 task_type 的成功率。"""
    buckets: dict[str, dict] = {}
    for task in task_types + ["overall"]:
        buckets[task] = {"total": 0, "hard": 0, "soft": 0.0}
    for r in results:
        tt = r.get("task_type", "other")
        for key in [tt, "overall"]:
            if key not in buckets:
                buckets[key] = {"total": 0, "hard": 0, "soft": 0.0}
            buckets[key]["total"] += 1
            buckets[key]["hard"] += int(r.get("hard", 0))
            buckets[key]["soft"] += float(r.get("soft", 0.0))
    return buckets


def _format_rejection_buffer(buffer: list[dict]) -> str:
    """**已弃用** — 为向后兼容保留；请使用 _format_step_buffer。"""
    return _format_step_buffer(buffer)


def _extract_failure_patterns(
    rollout_results: list[dict],
    step_dir: str,
) -> list[dict]:
    """从 rollout 结果提取紧凑的失败模式。

    优先使用 minibatch patch 中 analyst 的 ``failure_summary``；
    否则回退到 ``fail_reason`` 前缀分组。
    """
    failures = [r for r in rollout_results if not r.get("hard")]
    if not failures:
        return []

    # 按 fail_reason 前缀分组
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in failures:
        reason = r.get("fail_reason", "unknown")
        prefix = reason.split(":")[0].strip() if ":" in reason else reason
        groups[prefix].append(r)

    # 尝试从 analyst patch 获取更丰富的描述
    analyst_descs: list[str] = []
    patch_globs = [
        os.path.join(step_dir, "patches", "minibatch_fail_*.json"),
        os.path.join(step_dir, "batch_*", "patches", "minibatch_fail_*.json"),
    ]
    seen_patch_files: set[str] = set()
    for pattern in patch_globs:
        for fname in sorted(glob.glob(pattern)):
            if fname in seen_patch_files:
                continue
            seen_patch_files.add(fname)
            try:
                with open(fname) as f:
                    patch = json.load(f)
                for fs in patch.get("failure_summary", []):
                    ft = fs.get("failure_type", "")
                    sd = fs.get("description", "")
                    analyst_descs.append(f"{ft}: {sd}" if sd else ft)
            except Exception:
                pass

    patterns = []
    desc_iter = iter(analyst_descs)
    for prefix, items in groups.items():
        desc = next(desc_iter, None) or prefix
        patterns.append({
            "pattern": desc,
            "count": len(items),
            "task_ids": [str(r.get("id", "?")) for r in items],
        })
    return patterns


def _format_step_buffer(buffer: list[dict]) -> str:
    """将统一步骤缓冲区格式化为单一上下文块。

    每条记录描述先前步骤发生的事：rollout 中观察到的失败模式，
    以及步骤被拒绝时尝试的具体编辑及分数下降。

    *buffer* 为空时返回空字符串。
    """
    if not buffer:
        return ""

    parts = [
        "Below is a summary of previous steps in this epoch. "
        "Use it to avoid repeating ineffective edits and to prioritise "
        "failure patterns that remain unsolved.\n"
    ]

    for entry in buffer:
        step = entry["step"]
        action = entry["action"]
        n_fail = entry.get("n_fail", 0)
        n_total = entry.get("n_total", "?")

        parts.append(f"### Step {step} — {action.upper()} ({n_fail}/{n_total} failed)")

        # 失败模式
        for p in entry.get("failure_patterns", []):
            ids = ", ".join(p["task_ids"][:3])
            parts.append(f'  - "{p["pattern"]}" (×{p["count"]}, tasks: {ids})')

        # 被拒绝的编辑（仅 reject 时存在）
        rejected = entry.get("rejected_edits", [])
        if rejected:
            score_before = entry.get("score_before", "?")
            score_after = entry.get("score_after", "?")
            parts.append(
                f"  Rejected edits (score {score_before} → {score_after}):"
            )
            for i, e in enumerate(rejected, 1):
                if e.get("op") is not None:
                    op = e.get("op", "?")
                    content = e.get("content", "")
                    target = e.get("target", "")
                    if target:
                        parts.append(f'    {i}. [{op}] target="{target[:80]}" → "{content}"')
                    else:
                        parts.append(f'    {i}. [{op}] "{content}"')
                else:
                    kind = e.get("type", "?")
                    title = e.get("title", "")
                    instruction = e.get("instruction", "")
                    parts.append(f'    {i}. [{kind}] "{title}" → "{instruction}"')

    return "\n".join(parts)


# ── 训练器 ────────────────────────────────────────────────────────────────────

class ReflACTTrainer:
    """主 ReflACT 训练循环。

    Parameters
    ----------
    cfg : dict
        配置字典，完整键列表见 ``configs/alfworld_default.yaml``。
    adapter : EnvAdapter
        环境适配器实例。
    """

    def __init__(self, cfg: dict, adapter: EnvAdapter) -> None:
        self.cfg = cfg
        self.adapter = adapter

    def train(self) -> dict:
        """执行完整 ReflACT 训练循环，返回 summary 字典。"""
        cfg = self.cfg
        adapter = self.adapter
        out_root = cfg["out_root"]
        os.makedirs(out_root, exist_ok=True)
        art = _artifact_layout(cfg)

        # ── 适配器初始化（一次性）────────────────────────────────────
        adapter.setup(cfg)
        dataloader = adapter.get_dataloader()

        def _build_train_env(batch: BatchSpec):
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size, batch.seed

        def _build_eval_env(split: str, env_num: int, seed: int):
            if dataloader is None:
                env_manager = adapter.build_eval_env(
                    env_num=env_num,
                    split=split,
                    seed=seed,
                    out_root=out_root,
                )
                actual_n = len(env_manager) if hasattr(env_manager, "__len__") else env_num
                return env_manager, actual_n

            batch = dataloader.build_eval_batch(
                env_num=env_num,
                split=split,
                seed=seed,
                out_root=out_root,
            )
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size

        # ── 配置模型 ─────────────────────────────────────────────────────
        backend = cfg.get("model_backend", "azure_openai")
        configure_azure_openai(
            endpoint=(
                cfg.get("azure_openai_endpoint")
                or cfg.get("azure_endpoint")
                or None
            ),
            api_version=(
                cfg.get("azure_openai_api_version")
                or cfg.get("azure_api_version")
                or None
            ),
            api_key=(
                cfg.get("azure_openai_api_key")
                or cfg.get("azure_api_key")
                or None
            ),
            auth_mode=cfg.get("azure_openai_auth_mode") or None,
            ad_scope=cfg.get("azure_openai_ad_scope") or None,
            managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
            optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
            optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
            optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
            optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
            optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
            optimizer_managed_identity_client_id=(
                cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
            ),
            target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
            target_api_version=cfg.get("target_azure_openai_api_version") or None,
            target_api_key=cfg.get("target_azure_openai_api_key") or None,
            target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
            target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
            target_managed_identity_client_id=(
                cfg.get("target_azure_openai_managed_identity_client_id") or None
            ),
        )
        optimizer_backend = cfg.get("optimizer_backend")
        target_backend = cfg.get("target_backend")
        if not optimizer_backend or not target_backend:
            if backend in {"claude", "claude_chat"}:
                optimizer_backend = optimizer_backend or "claude_chat"
                target_backend = target_backend or "claude_chat"
            elif backend in {"codex", "codex_exec"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "codex_exec"
            elif backend == "claude_code_exec":
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "claude_code_exec"
            elif backend in {"qwen", "qwen_chat"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "qwen_chat"
            else:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "openai_chat"
            cfg["optimizer_backend"] = optimizer_backend
            cfg["target_backend"] = target_backend
        set_optimizer_backend(optimizer_backend)
        set_target_backend(target_backend)
        set_optimizer_deployment(cfg["optimizer_model"])
        set_target_deployment(cfg["target_model"])
        configure_codex_exec(
            path=cfg.get("codex_exec_path", "codex"),
            sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
            profile=cfg.get("codex_exec_profile", ""),
            full_auto=cfg.get("codex_exec_full_auto", False),
            reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
            use_sdk=cfg.get("codex_exec_use_sdk", None),
            network_access=cfg.get("codex_exec_network_access", False),
            web_search=cfg.get("codex_exec_web_search", False),
            approval_policy=cfg.get("codex_exec_approval_policy", "never"),
        )
        configure_claude_code_exec(
            path=cfg.get("claude_code_exec_path", "claude"),
            profile=cfg.get("claude_code_exec_profile", ""),
            use_sdk=cfg.get("claude_code_exec_use_sdk", None),
            effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
            max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
        )
        configure_qwen_chat(
            base_url=cfg.get("qwen_chat_base_url") or None,
            api_key=cfg.get("qwen_chat_api_key") or None,
            temperature=cfg.get("qwen_chat_temperature"),
            timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
            max_tokens=cfg.get("qwen_chat_max_tokens"),
            enable_thinking=cfg.get("qwen_chat_enable_thinking"),
        )
        os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] = (
            "1"
            if target_backend == "codex_exec" and cfg.get("codex_trace_to_optimizer", False)
            else "0"
        )
        reasoning = cfg.get("reasoning_effort", "") or None
        set_reasoning_effort(reasoning)
        print(
            f"  [model config] backend={backend}  "
            f"optimizer={cfg['optimizer_model']} ({optimizer_backend})  "
            f"target={cfg['target_model']} ({target_backend})  "
            f"reasoning={reasoning or 'off'}"
        )

        # ── 初始化 Ray ───────────────────────────────────────────────────
        if adapter.requires_ray():
            try:
                import ray
            except ImportError as e:
                raise ImportError(
                    "This environment requires ray, but ray is not installed."
                ) from e

            if not ray.is_initialized():
                ray.init(num_gpus=0)

        # ── 加载初始 prompt ──────────────────────────────────────────────
        prompt_init_path = _resolve_prompt_init_path(cfg)
        if prompt_init_path and os.path.exists(prompt_init_path):
            with open(prompt_init_path) as f:
                prompt_init = f.read()
            print(f"  [initial prompt] {prompt_init_path} ({len(prompt_init)} chars)")
        else:
            prompt_init = ""
            print("  [initial prompt] no prompt_init file — starting from blank")

        # ── 训练参数 ─────────────────────────────────────────────────────
        batch_size = cfg["batch_size"]
        num_epochs = cfg["num_epochs"]
        accumulation = cfg["accumulation"]
        seed = cfg["seed"]
        merge_bs = cfg["merge_batch_size"]
        max_analyst_rounds = int(cfg.get("max_analyst_rounds", 3) or 3)
        update_mode = normalize_update_mode(cfg.get("prompt_update_mode", "patch"))
        lr_control_mode = _normalise_lr_control_mode(cfg.get("lr_control_mode", "fixed"))
        if is_full_rewrite_minibatch_mode(update_mode):
            lr_control_mode = "none"
        longitudinal_pair_policy = _normalise_longitudinal_pair_policy(
            cfg.get("longitudinal_pair_policy", "mixed")
        )
        rewrite_reasoning_effort = cfg.get("rewrite_reasoning_effort", "high")
        if rewrite_reasoning_effort == "":
            rewrite_reasoning_effort = None
        rewrite_max_completion_tokens = int(cfg.get("rewrite_max_completion_tokens", 64000))
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if accumulation <= 0:
            raise ValueError(f"accumulation must be positive, got {accumulation}")

        train_size = _resolve_train_size(cfg, dataloader)
        steps_per_epoch = math.ceil(train_size / (batch_size * accumulation))
        batches_per_epoch = steps_per_epoch * accumulation
        total_steps = num_epochs * steps_per_epoch

        # 持久化派生字段，使 config.json / summary.json 与运行时一致
        # 与实际运行配方一致。
        cfg["train_size"] = train_size
        cfg["steps_per_epoch"] = steps_per_epoch
        cfg["batches_per_epoch"] = batches_per_epoch
        cfg["samples_per_epoch"] = train_size
        cfg["prompt_update_mode"] = update_mode
        cfg["lr_control_mode"] = lr_control_mode

        # 派生运行时值后保存配置。
        with open(os.path.join(out_root, "config.json"), "w") as f:
            json.dump(_redact_cfg(cfg), f, indent=2, ensure_ascii=False)

        train_pool_size = train_size

        scheduler = build_scheduler(
            mode=cfg.get("lr_scheduler", "constant"),
            max_lr=cfg["edit_budget"],
            min_lr=cfg.get("min_edit_budget", 2),
            total_steps=total_steps,
        )

        # 固定训练池：base_seeds（每个 seed 对应一个确定性 batch）
        if dataloader is not None:
            base_seeds = dataloader.make_base_seeds(
                steps_per_epoch=steps_per_epoch,
                accumulation=accumulation,
                seed=seed,
            )
        else:
            base_seeds = [seed + i + 1 for i in range(batches_per_epoch)]

        print(f"\n  [config] epochs={num_epochs} steps/epoch={steps_per_epoch} "
              f"(auto) accum={accumulation} batch_size={batch_size}")
        print(f"  [config] train_size={train_size}")
        print(f"  [config] batches/epoch={batches_per_epoch} "
              f"total_steps={total_steps} "
              f"games/epoch={train_pool_size}")
        print(f"  [config] lr_scheduler={cfg.get('lr_scheduler', 'constant')} "
              f"edit_budget={cfg['edit_budget']} "
              f"min_edit_budget={cfg.get('min_edit_budget', 2)}")
        print(f"  [config] prompt_update_mode={update_mode} "
              f"lr_control_mode={lr_control_mode} "
              f"rewrite_reasoning_effort={rewrite_reasoning_effort or 'off'} "
              f"rewrite_max_completion_tokens={rewrite_max_completion_tokens} "
              f"max_analyst_rounds={max_analyst_rounds}")
        print(f"  [config] longitudinal_pair_policy={longitudinal_pair_policy}")
        print(f"  [config] base_seeds={base_seeds}")

        # ── 断点续训检查 ─────────────────────────────────────────────────
        history = _load_history(out_root)
        runtime_state = _load_runtime_state(out_root)
        if runtime_state:
            last_step = int(runtime_state.get("last_completed_step", 0) or 0)
            current_prompt_path = (
                runtime_state.get("current_prompt_path")
                or runtime_state.get("current_skill_path")
                or _checkpoint_path(out_root, last_step, art)
            )
            with open(current_prompt_path) as f:
                current_prompt = f.read()
            best_prompt_path = (
                runtime_state.get("best_prompt_path")
                or runtime_state.get("best_skill_path")
                or _best_prompt_path(out_root, art)
            )
            if os.path.exists(best_prompt_path):
                with open(best_prompt_path) as f:
                    best_prompt = f.read()
            else:
                best_prompt = current_prompt
            current_score = float(runtime_state.get("current_score", -1.0) or -1.0)
            best_score = float(runtime_state.get("best_score", current_score) or current_score)
            best_step = runtime_state.get("best_step", last_step)
            current_origin = str(
                runtime_state.get("current_origin")
                or (f"step_{last_step:04d}" if last_step > 0 else "initial_prompt")
            )
            best_origin = str(runtime_state.get("best_origin") or current_origin)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f} "
                f"(origin={current_origin})"
            )
        elif history:
            last_step = history[-1]["step"]
            current_prompt = _load_prompt(out_root, last_step, art)
            best_rec = max(history, key=lambda h: h.get("best_score", 0.0))
            best_score = best_rec["best_score"]
            best_step = best_rec["best_step"]
            best_prompt_path = _best_prompt_path(out_root, art)
            if os.path.exists(best_prompt_path):
                with open(best_prompt_path) as f:
                    best_prompt = f.read()
            else:
                best_prompt = _load_prompt(out_root, best_step, art)
            current_score = history[-1].get("current_score", best_score)
            current_origin = f"step_{last_step:04d}"
            best_origin = f"step_{int(best_step):04d}" if isinstance(best_step, int) else str(best_step)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f}"
            )
        else:
            current_prompt = prompt_init
            best_prompt = prompt_init
            best_score = -1.0
            current_score = -1.0
            best_step = 0
            current_origin = "initial_prompt"
            best_origin = "initial_prompt"
            resume_from = 1

        _save_prompt(out_root, 0, prompt_init, art)

        def _persist_runtime_state(last_completed_step: int) -> None:
            best_path = os.path.join(out_root, art.best_file)
            _save_runtime_state(
                out_root,
                {
                    "last_completed_step": last_completed_step,
                    "current_prompt_path": _checkpoint_path(out_root, last_completed_step, art),
                    "current_score": current_score,
                    "current_origin": current_origin,
                    "best_prompt_path": best_path,
                    "best_score": best_score,
                    "best_step": best_step,
                    "best_origin": best_origin,
                },
            )

        # ── 筛选集缓存 ───────────────────────────────────────────────────
        sel_cache: dict[str, tuple[float, float]] = {}
        for rec in history:
            sh = rec.get("candidate_hash", "")
            if sh and rec.get("selection_hard") is not None:
                sel_cache[sh] = (rec["selection_hard"], rec["selection_soft"])

        # ── 筛选集基线评估 ───────────────────────────────────────────────
        if cfg.get("use_gate") is False:
            raise ValueError(
                "Gate validation is mandatory in this branch. Remove "
                "`evaluation.use_gate=false` from the config."
            )
        if current_score < 0:
            print(f"\n{'='*60}")
            print("  BASELINE — evaluate initial prompt on Selection set (valid_seen)")
            print(f"{'='*60}")
            sel_env, sel_n = _build_eval_env(
                split="valid_seen",
                env_num=cfg["sel_env_num"],
                seed=seed,
            )
            print(f"  Selection items: {sel_n}")
            baseline_dir = os.path.join(out_root, "selection_eval_baseline")
            baseline_results = adapter.rollout(sel_env, prompt_init, baseline_dir)
            current_score, baseline_soft = compute_score(baseline_results)
            best_score = current_score
            sh = prompt_hash(prompt_init)
            sel_cache[sh] = (current_score, baseline_soft)
            current_origin = "initial_prompt"
            best_origin = "initial_prompt"
            _persist_runtime_state(0)
            print(
                f"  [baseline result] selection hard={current_score:.4f} "
                f"soft={baseline_soft:.4f}"
            )

        # ── 训练循环 ─────────────────────────────────────────────────────
        t_loop_start = time.time()

        if resume_from > total_steps:
            print(f"\n  [skip] all {total_steps} steps complete — jumping to evaluation")

        global_step = 0
        for epoch in range(1, num_epochs + 1):
            if dataloader is not None:
                epoch_batches = dataloader.plan_train_epoch(
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                    accumulation=accumulation,
                    batch_size=batch_size,
                    seed=seed,
                    out_root=out_root,
                )
                shuffled_seeds = [batch.seed for batch in epoch_batches]
            else:
                epoch_batches = []
                epoch_rng = random.Random(seed + epoch * 1000)
                shuffled_seeds = base_seeds.copy()
                epoch_rng.shuffle(shuffled_seeds)

            # 步骤缓冲区：在本 epoch 内累积逐步上下文（失败模式 +
            # 被拒绝的编辑），供优化器看到完整历史。
            step_buffer: list[dict] = []
            active_meta_prompt = (
                _load_meta_prompt_content(out_root, epoch - 1)
                if cfg.get("use_meta_prompt", False)
                else ""
            )

            print(
                f"\n  [EPOCH {epoch}/{num_epochs}] "
                f"shuffled_seeds={shuffled_seeds}"
            )
            if active_meta_prompt:
                print(
                    f"  [meta prompt] loaded from epoch {epoch - 1} "
                    f"({len(active_meta_prompt)} chars)"
                )

            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                if global_step < resume_from:
                    continue

                step_t0 = time.time()
                step_dir = os.path.join(out_root, "steps", f"step_{global_step:04d}")
                os.makedirs(step_dir, exist_ok=True)

                tokens_before = get_token_summary()

                print(
                    f"\n  [STEP {global_step}/{total_steps}] "
                    f"epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"{'='*30}"
                )

                step_rec: dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "timing": {},
                    "tokens": {},
                }

                # ── 梯度累积：推演 + 反思 ────────────────────────────────
                all_failure_patches: list[dict] = []
                all_success_patches: list[dict] = []
                all_raw_patches: list[dict | None] = []
                all_rollout_results: list[dict] = []
                accum_rollout_stats: list[dict] = []
                total_rollout_time = 0.0
                total_reflect_time = 0.0

                for a in range(accumulation):
                    batch_idx = step_in_epoch * accumulation + a
                    if dataloader is not None:
                        batch_spec = epoch_batches[batch_idx]
                        train_env, train_n, batch_seed = _build_train_env(batch_spec)
                    else:
                        batch_seed = shuffled_seeds[batch_idx]
                        train_env = adapter.build_train_env(
                            batch_size=batch_size,
                            seed=batch_seed,
                            out_root=out_root,
                        )
                        train_n = len(train_env) if hasattr(train_env, "__len__") else batch_size

                    # 目录路由
                    if accumulation > 1:
                        batch_dir = os.path.join(step_dir, f"batch_{a}")
                    else:
                        batch_dir = step_dir

                    rollout_dir = os.path.join(batch_dir, "rollout")
                    patches_dir = os.path.join(batch_dir, "patches")

                    # ① 推演 ─────────────────────────────────────────────────
                    t_phase = time.time()
                    print(f"    [1/6 ROLLOUT] train items={train_n} (from pool, batch_seed={batch_seed})")
                    rollout_results = adapter.rollout(
                        train_env, current_prompt, rollout_dir,
                        use_eval_feedback=True,
                    )
                    r_hard, r_soft = compute_score(rollout_results)
                    total_rollout_time += time.time() - t_phase
                    all_rollout_results.extend(rollout_results)
                    print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")

                    # ② 反思 ─────────────────────────────────────────────────
                    t_phase = time.time()
                    pred_dir = os.path.join(rollout_dir, "predictions")

                    # 从缓冲区构建步骤上下文
                    step_buffer_context = _format_step_buffer(step_buffer)

                    raw_patches = adapter.reflect(
                        rollout_results, current_prompt, batch_dir,
                        prediction_dir=pred_dir, patches_dir=patches_dir,
                        random_seed=batch_seed,
                        step_buffer_context=step_buffer_context,
                        meta_prompt_context=active_meta_prompt,
                    )
                    failure_patches, success_patches = _normalise_patches(
                        raw_patches,
                        update_mode=update_mode,
                    )
                    all_failure_patches.extend(failure_patches)
                    all_success_patches.extend(success_patches)
                    all_raw_patches.extend(raw_patches)
                    total_reflect_time += time.time() - t_phase

                    print(
                        f"    [2/6 done] failure_patches={len(failure_patches)} "
                        f"success_patches={len(success_patches)}"
                    )

                    # 记录每 batch 统计
                    accum_rollout_stats.append({
                        "batch_idx": a,
                        "batch_seed": batch_seed,
                        "n_envs": len(rollout_results),
                        "hard": r_hard,
                        "soft": r_soft,
                        "n_failure_patches": len(failure_patches),
                        "n_success_patches": len(success_patches),
                    })

                # ── 梯度累积循环结束 ─────────────────────────────────────

                # 跨 batch 聚合 rollout 统计
                total_n = sum(b["n_envs"] for b in accum_rollout_stats)
                agg_hard = sum(b["hard"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)
                agg_soft = sum(b["soft"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)

                step_rec["rollout_hard"] = round(agg_hard, 6)
                step_rec["rollout_soft"] = round(agg_soft, 6)
                step_rec["rollout_n"] = total_n
                step_rec["accumulation_batches"] = accum_rollout_stats
                step_rec["timing"]["rollout_s"] = round(total_rollout_time, 1)
                step_rec["timing"]["reflect_s"] = round(total_reflect_time, 1)

                n_total_patches = len(all_failure_patches) + len(all_success_patches)
                step_rec["n_patches"] = n_total_patches
                step_rec["n_failure_patches"] = len(all_failure_patches)
                step_rec["n_success_patches"] = len(all_success_patches)

                if accumulation > 1:
                    print(
                        f"    [accum done] total: failure={len(all_failure_patches)} "
                        f"success={len(all_success_patches)} "
                        f"from {accumulation} batches"
                    )

                # ── 无 patch？跳过 ───────────────────────────────────────
                if not all_failure_patches and not all_success_patches:
                    step_rec["action"] = "skip_no_patches"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["prompt_len"] = len(current_prompt)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_prompt(out_root, global_step, current_prompt, art)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable patches — prompt unchanged")
                    continue

                # ③ 聚合 ─────────────────────────────────────────────────
                t_phase = time.time()
                merged_patch = merge_patches(
                    current_prompt, all_failure_patches, all_success_patches,
                    batch_size=merge_bs, verbose=True,
                    workers=cfg["analyst_workers"],
                    update_mode=update_mode,
                    meta_prompt_context=active_meta_prompt,
                )
                with open(os.path.join(step_dir, "merged_patch.json"), "w") as f:
                    json.dump(merged_patch, f, ensure_ascii=False, indent=2)

                merged_items = get_payload_items(merged_patch, update_mode)
                n_edits_merged = len(merged_items)
                step_rec["n_edits_merged"] = n_edits_merged
                step_rec["timing"]["aggregate_s"] = round(time.time() - t_phase, 1)
                print(f"    [3/6 done] merged {n_edits_merged} {payload_label(update_mode)}")

                # ④ 筛选 ─────────────────────────────────────────────────
                t_phase = time.time()
                lr_decision = None
                if is_full_rewrite_minibatch_mode(update_mode):
                    edit_budget = None
                    ranked_patch = merged_patch
                    ranked_items = merged_items
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["edit_budget"] = None
                    step_rec["lr_control_mode"] = "none"
                    with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                else:
                    if lr_control_mode == "autonomous":
                        lr_decision = decide_autonomous_learning_rate(
                            prompt_content=current_prompt,
                            merged_patch=merged_patch,
                            update_mode=update_mode,
                            rollout_hard=agg_hard,
                            rollout_soft=agg_soft,
                            rollout_n=total_n,
                            step_buffer_context=step_buffer_context,
                            meta_prompt_context=active_meta_prompt,
                        )
                        edit_budget = int(lr_decision["learning_rate"])
                        with open(os.path.join(step_dir, "lr_decision.json"), "w") as f:
                            json.dump(lr_decision, f, ensure_ascii=False, indent=2)
                        with open(os.path.join(out_root, "lr_history.jsonl"), "a") as f:
                            f.write(json.dumps({
                                "step": global_step,
                                "epoch": epoch,
                                **lr_decision,
                            }, ensure_ascii=False) + "\n")
                    else:
                        edit_budget = scheduler.step()
                    ranked_patch = rank_and_select(
                        current_prompt, merged_patch,
                        max_edits=edit_budget,
                        update_mode=update_mode,
                        meta_prompt_context=active_meta_prompt,
                    )
                    with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)

                    ranked_items = get_payload_items(ranked_patch, update_mode)
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["edit_budget"] = edit_budget
                    step_rec["lr_control_mode"] = lr_control_mode
                    if lr_decision is not None:
                        step_rec["lr_decision"] = lr_decision
                step_rec["timing"]["select_s"] = round(time.time() - t_phase, 1)

                support_counts = [
                    item.get("support_count", 0) for item in ranked_items if isinstance(item, dict)
                ]
                step_rec["support_counts"] = support_counts
                if is_full_rewrite_minibatch_mode(update_mode):
                    print(
                        f"    [4/6 SELECT] skipped LR/select; "
                        f"using {n_edits_ranked} merged {payload_label(update_mode)}"
                    )
                else:
                    print(
                        f"    [4/6 SELECT] "
                        f"{n_edits_merged} -> {n_edits_ranked} {payload_label(update_mode)} "
                        f"(budget={edit_budget}, lr_control={lr_control_mode})"
                    )

                # ⑤ 更新 ─────────────────────────────────────────────────
                t_phase = time.time()
                rewrite_result = None
                if update_mode == "rewrite_from_suggestions":
                    rewrite_result = rewrite_prompt_from_suggestions(
                        current_prompt,
                        ranked_patch,
                        step_buffer_context=step_buffer_context,
                        env=cfg.get("env"),
                        reasoning_effort=rewrite_reasoning_effort,
                        max_completion_tokens=rewrite_max_completion_tokens,
                    )
                    if rewrite_result and rewrite_result.get("new_prompt"):
                        candidate_prompt = rewrite_result["new_prompt"]
                        apply_report = []
                        with open(os.path.join(step_dir, "rewrite_result.json"), "w") as f:
                            json.dump(rewrite_result, f, ensure_ascii=False, indent=2)
                    else:
                        candidate_prompt = current_prompt
                        apply_report = []
                elif is_full_rewrite_minibatch_mode(update_mode):
                    prompt_candidates = get_payload_items(ranked_patch, update_mode)
                    selected_candidate = next(
                        (
                            item for item in prompt_candidates
                            if isinstance(item, dict) and str(item.get("new_prompt", "")).strip()
                        ),
                        None,
                    )
                    if selected_candidate:
                        candidate_prompt = str(selected_candidate["new_prompt"]).rstrip() + "\n"
                        apply_report = []
                        rewrite_result = {
                            "reasoning": ranked_patch.get("reasoning", ""),
                            "change_summary": selected_candidate.get("change_summary", []),
                            "title": selected_candidate.get("title", ""),
                            "source_type": selected_candidate.get("source_type", ""),
                        }
                        with open(os.path.join(step_dir, "full_rewrite_result.json"), "w") as f:
                            json.dump(
                                {
                                    "selected_candidate": selected_candidate,
                                    "merged_patch": ranked_patch,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    else:
                        candidate_prompt = current_prompt
                        apply_report = []
                else:
                    candidate_prompt, apply_report = apply_patch_with_report(current_prompt, ranked_patch)
                with open(os.path.join(step_dir, "candidate_prompt.md"), "w") as f:
                    f.write(candidate_prompt)
                if apply_report:
                    with open(os.path.join(step_dir, "edit_apply_report.json"), "w") as f:
                        json.dump(apply_report, f, indent=2, ensure_ascii=False)

                cand_hash = prompt_hash(candidate_prompt)
                step_rec["candidate_hash"] = cand_hash
                step_rec["candidate_prompt_len"] = len(candidate_prompt)
                if rewrite_result:
                    step_rec["rewrite_change_summary"] = rewrite_result.get("change_summary", [])
                if apply_report:
                    step_rec["edit_apply_summary"] = {
                        "total": len(apply_report),
                        "applied": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("applied")
                        ),
                        "skipped": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("skipped")
                        ),
                        "errors": sum(
                            1 for row in apply_report if row.get("status") == "error"
                        ),
                    }
                step_rec["timing"]["update_s"] = round(time.time() - t_phase, 1)
                if (
                    update_mode == "rewrite_from_suggestions"
                    and rewrite_result is None
                ) or (
                    is_full_rewrite_minibatch_mode(update_mode)
                    and rewrite_result is None
                ):
                    step_rec["action"] = "skip_no_rewrite"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["prompt_len"] = len(current_prompt)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_prompt(out_root, global_step, current_prompt, art)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable rewrite generated — prompt unchanged")
                    continue
                print(
                    f"    [5/6 UPDATE] "
                    f"prompt_len {len(current_prompt)} -> {len(candidate_prompt)}"
                )

                # ⑥ 评估 ─────────────────────────────────────────────────
                t_phase = time.time()
                if cand_hash in sel_cache:
                    cand_hard, cand_soft = sel_cache[cand_hash]
                    print(
                        f"    [6/6 EVALUATE] "
                        f"cache hit {cand_hash}: hard={cand_hard:.4f}"
                    )
                else:
                    sel_env, sel_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    print(f"    [6/6 EVALUATE] selection items={sel_n}")
                    sel_eval_dir = os.path.join(step_dir, "selection_eval")
                    sel_results = adapter.rollout(sel_env, candidate_prompt, sel_eval_dir)
                    cand_hard, cand_soft = compute_score(sel_results)
                    sel_cache[cand_hash] = (cand_hard, cand_soft)

                step_rec["selection_hard"] = cand_hard
                step_rec["selection_soft"] = cand_soft

                gate = evaluate_gate(
                    candidate_prompt=candidate_prompt,
                    cand_hard=cand_hard,
                    current_prompt=current_prompt,
                    current_score=current_score,
                    best_prompt=best_prompt,
                    best_score=best_score,
                    best_step=best_step,
                    global_step=global_step,
                )
                step_rec["action"] = gate.action
                prev_current = current_score
                prev_best = best_score
                current_prompt = gate.current_prompt
                current_score = gate.current_score
                best_prompt = gate.best_prompt
                best_score = gate.best_score
                best_step = gate.best_step
                if gate.action in {"accept", "accept_new_best"}:
                    current_origin = f"step_{global_step:04d}"
                if gate.action == "accept_new_best":
                    best_origin = current_origin

                if gate.action == "accept_new_best":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT (new best) "
                        f"hard={cand_hard:.4f} > prev best {prev_best:.4f}"
                    )
                elif gate.action == "accept":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT "
                        f"hard={cand_hard:.4f} > current={prev_current:.4f}"
                    )
                else:
                    print(
                        f"    [6/6 EVALUATE] REJECT "
                        f"hard={cand_hard:.4f} <= current={current_score:.4f}"
                    )

                step_rec["timing"]["evaluate_s"] = round(time.time() - t_phase, 1)

                # ── 步骤缓冲区：统一失败模式 + 被拒绝的编辑 ───────────────
                action = step_rec.get("action", "unknown")
                n_total = len(all_rollout_results) or 1
                n_fail = sum(1 for r in all_rollout_results if not r.get("hard"))
                failure_patterns = _extract_failure_patterns(
                    all_rollout_results, step_dir,
                )

                buf_entry: dict = {
                    "step": global_step,
                    "action": action,
                    "n_total": n_total,
                    "n_fail": n_fail,
                    "failure_patterns": failure_patterns,
                }

                # 步骤被拒绝时附加被拒绝的编辑
                if "reject" in action and ranked_patch:
                    rejected_edits = [
                        short_item_summary(item, update_mode)
                        for item in ranked_items
                        if isinstance(item, dict)
                    ]
                    buf_entry["score_before"] = current_score
                    buf_entry["score_after"] = cand_hard
                    buf_entry["rejected_edits"] = rejected_edits

                step_buffer.append(buf_entry)

                # 持久化步骤摘要，供步骤缓冲区上下文使用
                digest_path = os.path.join(step_dir, "trajectory_digest.json")
                with open(digest_path, "w") as f:
                    json.dump(buf_entry, f, indent=2, ensure_ascii=False)

                # ── Token 快照 ───────────────────────────────────────────
                tokens_after = get_token_summary()
                step_tokens: dict = {}
                for stage in tokens_after:
                    if stage == "_total":
                        continue
                    after = tokens_after[stage]
                    before = tokens_before.get(stage, {})
                    step_tokens[stage] = {
                        "calls": after.get("calls", 0) - before.get("calls", 0),
                        "prompt_tokens": after.get("prompt_tokens", 0)
                        - before.get("prompt_tokens", 0),
                        "completion_tokens": after.get("completion_tokens", 0)
                        - before.get("completion_tokens", 0),
                    }
                step_rec["tokens"] = step_tokens

                # ── 保存状态 ─────────────────────────────────────────────
                step_rec["current_score"] = current_score
                step_rec["best_score"] = best_score
                step_rec["best_step"] = best_step
                step_rec["current_origin"] = current_origin
                step_rec["best_origin"] = best_origin
                step_rec["prompt_len"] = len(current_prompt)
                step_rec["wall_time_s"] = round(time.time() - step_t0, 1)

                _save_prompt(out_root, global_step, current_prompt, art)
                with open(os.path.join(out_root, art.best_file), "w") as f:
                    f.write(best_prompt)
                history.append(step_rec)
                _save_history(out_root, history)
                _persist_runtime_state(global_step)
                with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                    json.dump(step_rec, f, indent=2, ensure_ascii=False)

                timing = step_rec["timing"]
                print(
                    f"\n  [STEP {global_step} done] "
                    f"epoch={epoch} action={step_rec['action']} "
                    f"current={current_score:.4f} best={best_score:.4f} "
                    f"dt={step_rec['wall_time_s']}s\n"
                    f"    timing: rollout={timing.get('rollout_s',0)}s "
                    f"reflect={timing.get('reflect_s',0)}s "
                    f"aggregate={timing.get('aggregate_s',0)}s "
                    f"select={timing.get('select_s',0)}s "
                    f"evaluate={timing.get('evaluate_s',0)}s"
                )

            epoch_last_step_prompt = current_prompt
            epoch_comparison_pairs: list[dict] | None = None

            # ── 慢更新（epoch 末）────────────────────────────────────────
            use_slow = cfg.get("use_slow_update", False)
            if use_slow:
                slow_dir = os.path.join(out_root, "slow_update", f"epoch_{epoch:02d}")
                slow_done_path = os.path.join(slow_dir, "slow_result.json")

                if os.path.exists(slow_done_path):
                    # 断点续训支持
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"resumed — already done"
                    )
                    with open(slow_done_path) as f:
                        slow_saved = json.load(f)
                    comparison_path = os.path.join(slow_dir, "comparison_pairs.json")
                    if os.path.exists(comparison_path):
                        try:
                            with open(comparison_path) as f:
                                epoch_comparison_pairs = json.load(f)
                        except Exception:
                            epoch_comparison_pairs = None
                    if (
                        slow_saved.get("slow_update_content")
                        and slow_saved.get("action") in {
                            "accept", "accept_new_best", "force_accept",
                        }
                        and epoch >= 2
                    ):
                        current_prompt = replace_slow_update_field(
                            current_prompt, slow_saved["slow_update_content"],
                        )
                        best_prompt = replace_slow_update_field(
                            best_prompt, slow_saved["slow_update_content"],
                        )
                elif epoch == 1:
                    # Epoch 1：注入空占位符
                    os.makedirs(slow_dir, exist_ok=True)
                    current_prompt = inject_empty_slow_update_field(current_prompt)
                    current_origin = f"slow_update_placeholder_epoch_{epoch:02d}"
                    _save_prompt(out_root, global_step, current_prompt, art)
                    with open(os.path.join(out_root, art.best_file), "w") as f:
                        f.write(best_prompt if best_score > current_score else current_prompt)
                    with open(slow_done_path, "w") as f:
                        json.dump({"action": "inject_placeholder", "epoch": epoch}, f, indent=2)
                    _persist_runtime_state(global_step)
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"injected empty placeholder"
                    )
                else:
                    # Epoch 2+：纵向对比
                    os.makedirs(slow_dir, exist_ok=True)
                    print(
                        f"\n  {'='*60}\n"
                        f"  SLOW UPDATE — Epoch {epoch} "
                        f"(comparing epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    # 1. 取上一 epoch 最后一步的 prompt
                    prev_epoch_records = [
                        h for h in history if h.get("epoch") == epoch - 1
                    ]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_prompt = _load_prompt(out_root, prev_epoch_last_step, art)

                    # 2. 从训练集采样条目
                    slow_n = cfg.get("slow_update_samples", 20)
                    slow_seed = seed + epoch * 2000
                    if dataloader is not None:
                        slow_batch = dataloader.build_train_batch(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                        slow_env = adapter.build_env_from_batch(
                            slow_batch, out_root=out_root,
                        )
                    else:
                        slow_env = adapter.build_train_env(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                    slow_items = list(slow_env) if hasattr(slow_env, "__iter__") else slow_env
                    print(f"    [slow update] sampled {len(slow_items)} train items (seed={slow_seed})")

                    # 3. 用两个 prompt 做 rollout
                    t_slow = time.time()
                    prev_rollout_dir = os.path.join(slow_dir, "rollout_prev")
                    curr_rollout_dir = os.path.join(slow_dir, "rollout_curr")
                    results_prev = adapter.rollout(slow_env, prev_prompt, prev_rollout_dir)
                    results_curr = adapter.rollout(slow_env, current_prompt, curr_rollout_dir)

                    prev_hard, _ = compute_score(results_prev)
                    curr_hard, _ = compute_score(results_curr)
                    print(
                        f"    [slow update] prev epoch hard={prev_hard:.4f}  "
                        f"curr epoch hard={curr_hard:.4f}"
                    )

                    # 4. 构建并保存结构化对比对
                    comparison_pairs, all_comparison_pairs = _build_longitudinal_pairs(
                        adapter=adapter,
                        dataloader=dataloader,
                        prev_prompt=prev_prompt,
                        curr_prompt=current_prompt,
                        initial_items=slow_items,
                        initial_prev_results=results_prev,
                        initial_curr_results=results_curr,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        policy=longitudinal_pair_policy,
                        target_n=slow_n,
                        seed=slow_seed,
                        out_root=out_root,
                    )
                    epoch_comparison_pairs = comparison_pairs
                    if all_comparison_pairs is not comparison_pairs:
                        save_comparison_pairs(
                            all_comparison_pairs,
                            os.path.join(slow_dir, "comparison_pairs_all.json"),
                        )
                    save_comparison_pairs(
                        comparison_pairs,
                        os.path.join(slow_dir, "comparison_pairs.json"),
                    )
                    n_regressed = sum(1 for p in comparison_pairs if p["category"] == "regressed")
                    n_improved = sum(1 for p in comparison_pairs if p["category"] == "improved")
                    n_persist = sum(1 for p in comparison_pairs if p["category"] == "persistent_fail")
                    n_stable = sum(1 for p in comparison_pairs if p["category"] == "stable_success")
                    print(
                        f"    [slow update] comparison: "
                        f"regressed={n_regressed} improved={n_improved} "
                        f"persistent_fail={n_persist} stable_success={n_stable} "
                        f"policy={longitudinal_pair_policy} "
                        f"kept={len(comparison_pairs)}/{len(all_comparison_pairs)}"
                    )

                    # 5. 提取上一轮 slow update 指导供反思
                    existing_guidance = extract_slow_update_field(current_prompt)

                    # 6. 优化器分析（结合上一轮指导反思）
                    slow_result = run_slow_update(
                        current_prompt,
                        results_prev,
                        results_curr,
                        slow_items,
                        prev_prompt=prev_prompt,
                        prev_slow_update_content=existing_guidance,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        comparison_pairs=comparison_pairs,
                    )
                    slow_time = round(time.time() - t_slow, 1)

                    if slow_result and slow_result.get("slow_update_content"):
                        slow_candidate = replace_slow_update_field(
                            current_prompt, slow_result["slow_update_content"],
                        )
                        slow_candidate_hash = prompt_hash(slow_candidate)
                        with open(os.path.join(slow_dir, "candidate_prompt.md"), "w") as f:
                            f.write(slow_candidate)
                        slow_result["time_s"] = slow_time
                        slow_result["prev_hard"] = prev_hard
                        slow_result["curr_hard"] = curr_hard
                        slow_result["candidate_hash"] = slow_candidate_hash
                        slow_result["update_origin"] = "slow_update_momentum"
                        slow_result["update_target"] = (
                            "Address longitudinal regressions and persistent failures "
                            "observed across adjacent epochs."
                        )

                        # slow update 字段无条件强制写入 current_prompt 与 best_prompt；
                        # epoch 级纵向指导应始终保留，不得被步骤级筛选分数门控。
                        slow_content = slow_result["slow_update_content"]
                        current_prompt = replace_slow_update_field(
                            current_prompt, slow_content,
                        )
                        best_prompt = replace_slow_update_field(
                            best_prompt, slow_content,
                        )
                        # 更新缓存，使后续步骤使用
                        # 已注入 slow update 的 prompt 做哈希。
                        slow_candidate_hash = prompt_hash(current_prompt)
                        sel_cache[slow_candidate_hash] = (current_score, 0.0)

                        slow_result["action"] = "force_accept"
                        current_origin = f"slow_update_epoch_{epoch:02d}"

                        print(
                            f"    [slow update] force-injected into current & best "
                            f"({len(slow_content)} chars), "
                            f"{slow_time}s"
                        )
                    else:
                        slow_result = slow_result or {}
                        slow_result["action"] = "no_content"
                        slow_result["time_s"] = slow_time
                        print(
                            f"    [slow update] no guidance produced, "
                            f"{slow_time}s"
                        )

                    # 5. 保存
                    with open(slow_done_path, "w") as f:
                        json.dump(slow_result, f, indent=2, ensure_ascii=False)
                    _save_prompt(out_root, global_step, current_prompt, art)
                    with open(os.path.join(out_root, art.best_file), "w") as f:
                        f.write(best_prompt)
                    _persist_runtime_state(global_step)

                    print(
                        f"\n  [SLOW UPDATE epoch {epoch} done] "
                        f"current={current_score:.4f} best={best_score:.4f}"
                    )

            # ── 元 Skill（epoch 末，优化器侧记忆）────────────────────────
            use_meta_prompt = cfg.get("use_meta_prompt", False)
            if use_meta_prompt:
                meta_prompt_dir = os.path.join(out_root, "meta_prompt", f"epoch_{epoch:02d}")
                meta_prompt_done_path = os.path.join(meta_prompt_dir, "meta_prompt_result.json")
                os.makedirs(meta_prompt_dir, exist_ok=True)

                if os.path.exists(meta_prompt_done_path):
                    print(f"\n  [META SKILL epoch {epoch}] resumed — already done")
                elif epoch == 1:
                    with open(meta_prompt_done_path, "w") as f:
                        json.dump(
                            {"action": "skip_first_epoch", "epoch": epoch},
                            f, indent=2, ensure_ascii=False,
                        )
                    print(f"\n  [META SKILL epoch {epoch}] skipped — first epoch")
                else:
                    print(
                        f"\n  {'='*60}\n"
                        f"  META SKILL — Epoch {epoch} "
                        f"(optimizer memory from epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    prev_epoch_records = [h for h in history if h.get("epoch") == epoch - 1]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_prompt = _load_prompt(out_root, prev_epoch_last_step, art)
                    prev_meta_prompt = _load_meta_prompt_content(out_root, epoch - 1)

                    if epoch_comparison_pairs is None:
                        meta_n = cfg.get("slow_update_samples", 20)
                        meta_seed = seed + epoch * 2000
                        if dataloader is not None:
                            meta_batch = dataloader.build_train_batch(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                            meta_env = adapter.build_env_from_batch(
                                meta_batch, out_root=out_root,
                            )
                        else:
                            meta_env = adapter.build_train_env(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                        meta_items = list(meta_env) if hasattr(meta_env, "__iter__") else meta_env
                        prev_rollout_dir = os.path.join(meta_prompt_dir, "rollout_prev")
                        curr_rollout_dir = os.path.join(meta_prompt_dir, "rollout_curr")
                        results_prev = adapter.rollout(meta_env, prev_prompt, prev_rollout_dir)
                        results_curr = adapter.rollout(meta_env, epoch_last_step_prompt, curr_rollout_dir)
                        epoch_comparison_pairs, all_meta_comparison_pairs = _build_longitudinal_pairs(
                            adapter=adapter,
                            dataloader=dataloader,
                            prev_prompt=prev_prompt,
                            curr_prompt=epoch_last_step_prompt,
                            initial_items=meta_items,
                            initial_prev_results=results_prev,
                            initial_curr_results=results_curr,
                            prev_rollout_dir=prev_rollout_dir,
                            curr_rollout_dir=curr_rollout_dir,
                            policy=longitudinal_pair_policy,
                            target_n=meta_n,
                            seed=meta_seed,
                            out_root=out_root,
                        )
                        if all_meta_comparison_pairs is not epoch_comparison_pairs:
                            save_comparison_pairs(
                                all_meta_comparison_pairs,
                                os.path.join(meta_prompt_dir, "comparison_pairs_all.json"),
                            )
                        save_comparison_pairs(
                            epoch_comparison_pairs,
                            os.path.join(meta_prompt_dir, "comparison_pairs.json"),
                        )
                        meta_counts = _pair_category_counts(epoch_comparison_pairs)
                        print(
                            f"    [meta prompt] comparison: "
                            f"regressed={meta_counts.get('regressed', 0)} "
                            f"improved={meta_counts.get('improved', 0)} "
                            f"persistent_fail={meta_counts.get('persistent_fail', 0)} "
                            f"stable_success={meta_counts.get('stable_success', 0)} "
                            f"policy={longitudinal_pair_policy} "
                            f"kept={len(epoch_comparison_pairs)}/{len(all_meta_comparison_pairs)}"
                        )

                    t_meta_prompt = time.time()
                    meta_prompt_result = run_meta_prompt(
                        prev_prompt=prev_prompt,
                        curr_prompt=epoch_last_step_prompt,
                        comparison_pairs=epoch_comparison_pairs or [],
                        prev_meta_prompt_content=prev_meta_prompt,
                    )
                    meta_prompt_time = round(time.time() - t_meta_prompt, 1)

                    if meta_prompt_result and meta_prompt_result.get("meta_prompt_content"):
                        meta_prompt_result["time_s"] = meta_prompt_time
                        meta_prompt_result["action"] = "write_meta_prompt"
                        print(
                            f"    [meta prompt] memory written "
                            f"({len(meta_prompt_result['meta_prompt_content'])} chars), "
                            f"{meta_prompt_time}s"
                        )
                    else:
                        meta_prompt_result = meta_prompt_result or {}
                        meta_prompt_result["time_s"] = meta_prompt_time
                        meta_prompt_result["action"] = "no_content"
                        print(f"    [meta prompt] no memory produced, {meta_prompt_time}s")

                    with open(meta_prompt_done_path, "w") as f:
                        json.dump(meta_prompt_result, f, indent=2, ensure_ascii=False)

        # ── 保存最佳 prompt ───────────────────────────────────────────────
        with open(os.path.join(out_root, art.best_file), "w") as f:
            f.write(best_prompt)
        _persist_runtime_state(global_step)
        print(
            f"\n  [done] best prompt from step {best_step}, "
            f"score={best_score:.4f}"
        )

        # ── 最终测试评估（valid_unseen）──────────────────────────────────
        baseline_test_hard = None
        baseline_test_soft = None
        test_hard = None
        test_soft = None

        if cfg["eval_test"]:
            task_types = adapter.get_task_types()

            # 基线：测试集上的 S_0（valid_unseen）
            print(f"\n{'='*60}")
            print("  BASELINE TEST — evaluate initial prompt on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env, test_n = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n}")
            baseline_test_dir = os.path.join(out_root, "test_eval_baseline")
            baseline_test_results = adapter.rollout(test_env, prompt_init, baseline_test_dir)
            baseline_test_hard, baseline_test_soft = compute_score(baseline_test_results)
            baseline_buckets = _compute_task_type_buckets(baseline_test_results, task_types)
            print("\n  === Baseline Test Results (S_0) ===")
            for task_type in task_types + ["overall"]:
                b = baseline_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(baseline_test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in baseline_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # 测试集上的最佳 prompt
            print(f"\n{'='*60}")
            print("  BEST PROMPT TEST — evaluate best prompt on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env2, test_n2 = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n2}")
            test_dir = os.path.join(out_root, "test_eval")
            test_results = adapter.rollout(test_env2, best_prompt, test_dir)
            test_hard, test_soft = compute_score(test_results)
            best_buckets = _compute_task_type_buckets(test_results, task_types)
            print("\n  === Best Skill Test Results ===")
            for task_type in task_types + ["overall"]:
                b = best_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in best_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # 对比
            delta_hard = (test_hard or 0) - (baseline_test_hard or 0)
            print(f"\n  === Improvement (best vs baseline) ===")
            print(
                f"    hard: {baseline_test_hard:.4f} -> {test_hard:.4f}  "
                f"(delta={delta_hard:+.4f})"
            )

        # ── 全局摘要 ─────────────────────────────────────────────────────
        total_wall = time.time() - t_loop_start
        n_accept = sum(1 for h in history if "accept" in h.get("action", ""))
        n_reject = sum(1 for h in history if h.get("action") == "reject")
        n_skip = sum(1 for h in history if h.get("action") == "skip_no_patches")

        token_summary = get_token_summary()

        # Epoch 级统计
        epoch_stats = []
        for e in range(1, num_epochs + 1):
            epoch_records = [h for h in history if h.get("epoch") == e]
            if epoch_records:
                epoch_stats.append({
                    "epoch": e,
                    "steps": [h["step"] for h in epoch_records],
                    "accepts": sum(1 for h in epoch_records if "accept" in h.get("action", "")),
                    "rejects": sum(1 for h in epoch_records if h.get("action") == "reject"),
                    "skips": sum(1 for h in epoch_records if h.get("action") == "skip_no_patches"),
                    "best_score_at_epoch_end": epoch_records[-1].get("best_score", 0.0),
                    "current_score_at_epoch_end": epoch_records[-1].get("current_score", 0.0),
                })

        summary = {
            "version": "skillopt-0.1.0",
            "config": _redact_cfg(cfg),
            "baseline_selection_hard": sel_cache.get(
                prompt_hash(prompt_init), (None, None),
            )[0],
            "best_selection_hard": best_score,
            "best_step": best_step,
            "current_origin": current_origin,
            "best_origin": best_origin,
            "total_steps": len(history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_skips": n_skip,
            "epoch_stats": epoch_stats,
            "baseline_test_hard": baseline_test_hard,
            "baseline_test_soft": baseline_test_soft,
            "test_hard": test_hard,
            "test_soft": test_soft,
            "test_delta_hard": (
                (test_hard or 0) - (baseline_test_hard or 0)
                if test_hard is not None
                else None
            ),
            "total_wall_time_s": round(total_wall, 1),
            "token_summary": token_summary,
        }
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("  Final Summary")
        print(f"{'='*60}")
        print(
            f"  steps={len(history)} accept={n_accept} "
            f"reject={n_reject} skip={n_skip}"
        )
        print(f"  best_score={best_score:.4f} (step {best_step})  wall={total_wall:.0f}s")
        if epoch_stats:
            for es in epoch_stats:
                print(
                    f"    epoch {es['epoch']}: accept={es['accepts']} reject={es['rejects']} "
                    f"best={es['best_score_at_epoch_end']:.4f}"
                )
        if test_hard is not None:
            print(f"  test_hard={test_hard:.4f} test_soft={test_soft:.4f}")
        if token_summary.get("_total"):
            t = token_summary["_total"]
            print(
                f"  total tokens: {t['total_tokens']:,} "
                f"(prompt={t['prompt_tokens']:,} "
                f"completion={t['completion_tokens']:,} "
                f"calls={t['calls']})"
            )

        return summary
