"""【功能描述】Reflect 核心引擎 — minibatch trajectory 分析，产出局部 patch。

【输入】rollout 结果、当前 prompt、prediction_dir 等。

【输出】run_minibatch_reflect 返回含 source_type 的 patch dict 列表。
"""
from __future__ import annotations

import json
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from promptopt.model import chat_optimizer
from promptopt.optimizer.meta_prompt import format_meta_prompt_context
from promptopt.optimizer.update_modes import (
    get_payload_items,
    normalize_update_mode,
    payload_key,
    payload_label,
    truncate_payload,
)
from promptopt.templates import fill_prompt, has_prompt
from promptopt.utils import extract_json


def _optional_section(title: str, body: str) -> str:
    text = (body or "").strip()
    if not text:
        return ""
    return f"## {title}\n{text}\n\n"


def _build_flat_analyst_user(
    template_name: str,
    *,
    prompt_content: str,
    edit_budget: int,
    update_mode: str,
    trajectories_text: str,
    trajectory_count: int,
    step_buffer_context: str,
    meta_prompt_context: str,
) -> str:
    mode = normalize_update_mode(update_mode)
    ctx = (step_buffer_context or "").strip()
    meta = format_meta_prompt_context(meta_prompt_context)
    return fill_prompt(template_name, {
        "current_prompt": prompt_content,
        "edit_budget": str(edit_budget),
        "payload_label": payload_label(mode),
        "trajectory_count": str(trajectory_count),
        "trajectories": trajectories_text,
        "previous_steps_section": _optional_section("先前步骤摘要", ctx),
        "meta_section": meta.strip(),
    })


# ── Trajectory 格式化 ────────────────────────────────────────────────────

_MAX_TRAJ_CHARS = 12_000


def _clip_text(value, limit: int) -> str:
    """截断前安全渲染可选 trajectory 字段。"""
    if value is None:
        return ""
    return str(value)[:limit]


def fmt_trajectory(
    conversation: list[dict],
    max_chars: int = _MAX_TRAJ_CHARS,
) -> str:
    """将 conversation 列表格式化为 analyst 可读文本。

    支持两种常见格式：

    1. 工具调用记录：   ``{"type": "tool_call", "cmd": ..., "obs": ...}``
    2. 步骤记录：        ``{"step": N, "action": ..., "env_feedback": ..., "reasoning": ...}``

    其他 dict 通过 ``"content"`` 键渲染。
    """
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {_clip_text(item, 500)}")
            continue
        if item.get("type") == "tool_call":
            cmd = _clip_text(item.get("cmd"), 500)
            obs = _clip_text(item.get("obs"), 800)
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = _clip_text(item.get("reasoning"), 300)
            action = _clip_text(item.get("action"), 200)
            feedback = _clip_text(item.get("env_feedback"), 500)
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif item.get("role") == "system":
            # 执行后验证 / enrichment 信息
            msg = _clip_text(item.get("content"), 2000)
            lines.append(f"[verification] {msg}")
        else:
            msg = _clip_text(item.get("content"), 500)
            role = item.get("role", "agent")
            lines.append(f"[{role}] {msg}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n...[middle truncated]...\n" + tail
    return text


# ── Minibatch trajectory 格式化 ──────────────────────────────────────────


def fmt_minibatch_trajectories(
    items: list[dict],
    prediction_dir: str,
) -> str:
    """格式化多条 trajectory 供 minibatch analyst 消费。

    每个 item 为含 ``"id"``、``"task_description"``、``"task_type"``、
    ``"fail_reason"`` 等的 rollout 结果 dict。为每个 item 读取 ``conversation.json``
    并连同 trajectory 头一并格式化。

    若可用，包含 spreadsheet preview 与 target system prompt，
    以便 analyst 看到 agent 所见内容。

    Parameters
    ----------
    items : list[dict]
        属于同一 minibatch 的 rollout 结果 dict 列表。
    prediction_dir : str
        含 ``predictions/`` 目录的路径，其中有 ``<task_id>/conversation.json``。

    Returns
    -------
    str
        所有 trajectory 以 ``---`` 分隔的格式化文本。
    """
    parts: list[str] = []
    for idx, item in enumerate(items, 1):
        tid = str(item["id"])
        conv_path = os.path.join(prediction_dir, tid, "conversation.json")
        conversation = None
        if os.path.exists(conv_path):
            with open(conv_path, encoding="utf-8") as f:
                conversation = json.load(f)
        elif item.get("trajectory_text"):
            conversation = [
                {"type": "tool_call", "cmd": "rollout", "obs": str(item.get("trajectory_text"))},
            ]
        if not conversation:
            continue

        traj_text = fmt_trajectory(conversation)
        header = (
            f"### Trajectory {idx} (id={tid})\n"
            f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
            f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
        )
        fail_reason = item.get("fail_reason", "")
        if fail_reason:
            header += f"Failure reason: {fail_reason}\n"
        header += f"Steps: {item.get('n_turns', '?')}\n"

        reference_text = str(item.get("reference_text") or "").strip()
        if reference_text:
            header += (
                f"\n#### Hidden Reference\n"
                f"{reference_text[:4000]}\n"
            )

        # ── 追加 target 上下文（agent 所见） ──────────────
        target_prompt = item.get("target_system_prompt", "")
        if not target_prompt:
            prompt_path = os.path.join(prediction_dir, tid, "target_system_prompt.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    target_prompt = f.read()
        if target_prompt:
            header += (
                f"\n#### Target System Prompt\n"
                f"{target_prompt[:3000]}\n"
            )

        user_prompt = item.get("target_user_prompt", "")
        if not user_prompt:
            user_prompt_path = os.path.join(prediction_dir, tid, "target_user_prompt.txt")
            if os.path.exists(user_prompt_path):
                with open(user_prompt_path) as f:
                    user_prompt = f.read()
        if user_prompt:
            header += (
                f"\n#### Target User Prompt\n"
                f"{user_prompt[:3000]}\n"
            )

        if os.environ.get("REFLACT_CODEX_TRACE_TO_OPTIMIZER", "0") == "1":
            codex_trace_summary = item.get("codex_trace_summary", "")
            if not codex_trace_summary:
                codex_trace_summary_path = os.path.join(prediction_dir, tid, "codex_trace_summary.txt")
                if os.path.exists(codex_trace_summary_path):
                    with open(codex_trace_summary_path) as f:
                        codex_trace_summary = f.read()
            if codex_trace_summary:
                header += (
                    f"\n#### Codex Trace Summary\n"
                    f"{codex_trace_summary}\n"
                )

        codex_probe_trace_steps = str(item.get("codex_probe_trace_steps") or "").strip()
        if codex_probe_trace_steps:
            header += (
                f"\n#### Codex Trace Steps\n"
                f"{codex_probe_trace_steps}\n"
            )

        preview = item.get("spreadsheet_preview", "")
        if not preview:
            preview_path = os.path.join(prediction_dir, tid, "spreadsheet_preview.txt")
            if os.path.exists(preview_path):
                with open(preview_path) as f:
                    preview = f.read()
        if preview:
            header += (
                f"\n#### Spreadsheet Preview\n"
                f"{preview[:3000]}\n"
            )

        parts.append(header + "\n" + traj_text)

    return "\n\n---\n\n".join(parts)


# ── Prompt 解析 ───────────────────────────────────────────────────────


# ── Minibatch analyst ──────────────────────────────────────────────────────


def run_error_analyst_minibatch(
    prompt_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_prompt_context: str = "",
    update_mode: str = "patch",
) -> dict | None:
    """单次 optimizer 调用分析一组失败 trajectory 的 minibatch。

    Parameters
    ----------
    prompt_content : str
        当前 prompt 文档文本。
    items : list[dict]
        Rollout 结果 dict（均应 ``hard=0``）。
    prediction_dir : str
        ``predictions/`` 目录路径。
    edit_budget : int
        最大编辑数（L）。
    system_prompt : str | None
        自定义 system prompt。``None`` = 使用通用默认。
    rejection_context : str
        *已弃用* — 请使用 ``step_buffer_context``。
    trajectory_memory_context : str
        *已弃用* — 请使用 ``step_buffer_context``。
    step_buffer_context : str
        先前步骤的统一摘要（失败模式 + 被拒绝的编辑）。

    Returns
    -------
    dict | None
        含 ``source_type="failure"`` 的 patch dict，出错时 ``None``。
    """
    mode = normalize_update_mode(update_mode)
    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    ctx = step_buffer_context or rejection_context or ""
    if trajectory_memory_context:
        ctx = f"{ctx}\n{trajectory_memory_context}" if ctx else trajectory_memory_context

    if system_prompt is not None:
        raise ValueError("T2I 引擎使用 promptopt/prompts/ 模板，不支持自定义 system_prompt")
    if not has_prompt("reflect_analyst_error"):
        raise FileNotFoundError("缺少模板 promptopt/prompts/reflect_analyst_error.md")

    user = _build_flat_analyst_user(
        "reflect_analyst_error",
        prompt_content=prompt_content,
        edit_budget=edit_budget,
        update_mode=mode,
        trajectories_text=trajectories_text,
        trajectory_count=len(items),
        step_buffer_context=ctx,
        meta_prompt_context=meta_prompt_context,
    )

    try:
        response, _ = chat_optimizer(
            system="", user=user,
            max_completion_tokens=4096,
            retries=3,
            stage="analyst",
        )
        result = extract_json(response)
        if result and "patch" in result:
            result["source_type"] = "failure"
            truncate_payload(result["patch"], edit_budget, mode)
            return result
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return None


def run_success_analyst_minibatch(
    prompt_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_prompt_context: str = "",
    update_mode: str = "patch",
) -> dict | None:
    """单次 optimizer 调用分析一组成功 trajectory 的 minibatch。

    Parameters
    ----------
    system_prompt : str | None
        自定义 system prompt。``None`` = 使用通用默认。
    trajectory_memory_context : str
        *已弃用* — 请使用 ``step_buffer_context``。
    step_buffer_context : str
        先前步骤的统一摘要（失败模式 + 被拒绝的编辑）。

    Returns
    -------
    dict | None
        含 ``source_type="success"`` 的 patch dict，出错时 ``None``。
    """
    mode = normalize_update_mode(update_mode)
    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    ctx = step_buffer_context or trajectory_memory_context or ""

    if system_prompt is not None:
        raise ValueError("T2I 引擎使用 promptopt/prompts/ 模板，不支持自定义 system_prompt")
    if not has_prompt("reflect_analyst_success"):
        raise FileNotFoundError("缺少模板 promptopt/prompts/reflect_analyst_success.md")

    user = _build_flat_analyst_user(
        "reflect_analyst_success",
        prompt_content=prompt_content,
        edit_budget=edit_budget,
        update_mode=mode,
        trajectories_text=trajectories_text,
        trajectory_count=len(items),
        step_buffer_context=ctx,
        meta_prompt_context=meta_prompt_context,
    )

    try:
        response, _ = chat_optimizer(
            system="", user=user,
            max_completion_tokens=4096,
            retries=3,
            stage="analyst",
        )
        result = extract_json(response)
        if result and "patch" in result:
            result["source_type"] = "success"
            truncate_payload(result["patch"], edit_budget, mode)
            return result
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return None


# ── Minibatch reflect 调度器 ────────────────────────────────────────────


def _split_minibatches(items: list, batch_size: int) -> list[list]:
    """将 items 拆分为最多 *batch_size* 个一组的 minibatch。"""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _shuffle_for_minibatch(items: list, seed: int | None) -> list:
    """返回 minibatch 顺序下的 items。

    提供 seed 时使用确定性 shuffle，使 resume 运行保持相同 minibatch 组成。
    无 seed 时回退为输入顺序。
    """
    ordered = list(items)
    if seed is None:
        return ordered
    random.Random(seed).shuffle(ordered)
    return ordered


def run_minibatch_reflect(
    results: list[dict],
    prompt_content: str,
    prediction_dir: str,
    patches_dir: str,
    workers: int,
    failure_only: bool,
    minibatch_size: int = 8,
    edit_budget: int = 4,
    random_seed: int | None = None,
    *,
    error_system: str | None = None,
    success_system: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_prompt_context: str = "",
    update_mode: str = "patch",
) -> list[dict | None]:
    """完整 minibatch reflect 阶段：分组 → 并行 optimizer 调用 → patch。

    分离失败与成功 trajectory，各按大小 M 拆成 minibatch，
    并行运行所有 minibatch 并保存 patch 文件。

    Parameters
    ----------
    results : list[dict]
        Rollout 结果 dict；见 :class:`~promptopt.types.RolloutResult`。
    prompt_content : str
        当前 prompt 文档。
    prediction_dir : str
        含 ``conversation.json`` 的 ``predictions/`` 路径。
    patches_dir : str
        保存逐 minibatch patch JSON 的路径。
    workers : int
        最大并行 optimizer 调用数。
    failure_only : bool
        为 True 时跳过成功 trajectory。
    minibatch_size : int
        每组 trajectory 数（M）。
    edit_budget : int
        每个 minibatch 最大编辑数（L）。
    random_seed : int | None
        可选 seed，在 minibatch 拆分前 shuffle trajectory。
    error_system, success_system : str | None
        可选自定义 prompt。``None`` = 使用通用默认。

    Returns
    -------
    list[dict | None]
        含 ``source_type`` "failure" 或 "success" 的 patch dict 列表。
    """
    os.makedirs(patches_dir, exist_ok=True)

    # 分离失败 / 成功
    failures = [r for r in results if not r.get("hard")]
    successes = [r for r in results if r.get("hard")] if not failure_only else []

    failures = _shuffle_for_minibatch(failures, random_seed)
    successes = _shuffle_for_minibatch(successes, None if random_seed is None else random_seed + 1)

    # 拆成 minibatch
    fail_batches = _split_minibatches(failures, minibatch_size)
    succ_batches = _split_minibatches(successes, minibatch_size)

    n_fail_batches = len(fail_batches)
    n_succ_batches = len(succ_batches)
    print(
        f"    [2/6 REFLECT minibatch] "
        f"failure={len(failures)}→{n_fail_batches} groups  "
        f"success={len(successes)}→{n_succ_batches} groups  "
        f"(M={minibatch_size}, L={edit_budget}, workers={workers})"
    )

    raw_patches: list[dict | None] = []

    # Resume 支持：检查已完成的 minibatch patch
    pending_fail: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(fail_batches):
        path = os.path.join(patches_dir, f"minibatch_fail_{idx:03d}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw_patches.append(json.load(f))
        else:
            pending_fail.append((idx, batch))

    pending_succ: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(succ_batches):
        path = os.path.join(patches_dir, f"minibatch_succ_{idx:03d}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw_patches.append(json.load(f))
        else:
            pending_succ.append((idx, batch))

    # ── Worker 函数 ──────────────────────────────────────────────────
    def _do_fail(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_error_analyst_minibatch(
            prompt_content, batch, prediction_dir,
            edit_budget=edit_budget,
            system_prompt=error_system,
            step_buffer_context=step_buffer_context,
            # 向后兼容回退
            rejection_context=rejection_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_prompt_context=meta_prompt_context,
            update_mode=update_mode,
        )
        return f"minibatch_fail_{idx:03d}", patch

    def _do_succ(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_success_analyst_minibatch(
            prompt_content, batch, prediction_dir,
            edit_budget=edit_budget,
            system_prompt=success_system,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_prompt_context=meta_prompt_context,
            update_mode=update_mode,
        )
        return f"minibatch_succ_{idx:03d}", patch

    # 并行运行所有待处理 minibatch
    all_pending = (
        [("fail", idx, batch) for idx, batch in pending_fail]
        + [("succ", idx, batch) for idx, batch in pending_succ]
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for kind, idx, batch in all_pending:
            if kind == "fail":
                futs[ex.submit(_do_fail, idx, batch)] = (kind, idx, len(batch))
            else:
                futs[ex.submit(_do_succ, idx, batch)] = (kind, idx, len(batch))

        for i, fut in enumerate(as_completed(futs), 1):
            kind, idx, batch_len = futs[fut]
            tag, patch = fut.result()
            if patch:
                path = os.path.join(patches_dir, f"{tag}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(patch, f, ensure_ascii=False, indent=2)
                raw_patches.append(patch)
            n_edits = len(get_payload_items(patch.get("patch", {}) if patch else {}, update_mode))
            print(
                f"      [analyst] {i}/{len(all_pending)} {tag} "
                f"({batch_len} trajs) → {n_edits} {payload_label(update_mode)}"
            )

    return raw_patches
