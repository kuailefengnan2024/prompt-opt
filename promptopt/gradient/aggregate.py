"""【功能描述】Reflect Aggregate 阶段 — 层次化 patch 合并；将 Reflect 阶段独立生成的 patch 通过层次化 LLM 调用合并为单一连贯 patch，失败驱动 patch 优先于成功驱动 patch。

【输入】prompt_content、failure_patches、success_patches、batch_size、update_mode 等。

【输出】合并后的 Patch dict（含 edits/reasoning 或对应 payload 键）。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from promptopt.model import chat_optimizer
from promptopt.optimizer.meta_prompt import format_meta_prompt_context
from promptopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from promptopt.llm_templates import load_template
from promptopt.utils import extract_json


# ── 内部辅助函数 ──────────────────────────────────────────────────────────

def _merge_batch(
    prompt_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    meta_prompt_context: str = "",
    level: int = 1,
) -> dict:
    """调用 optimizer LLM 将一批 patch 合并为一个。"""
    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    user = (
        f"## Current Prompt\n{prompt_content}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n{patches_text}"
    )
    optimizer_ctx = format_meta_prompt_context(meta_prompt_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=system_prompt,
            user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(update_mode) else 4096,
            retries=3,
            stage="merge",
        )
        merged = extract_json(response)
        key = payload_key(update_mode)
        if merged and key in merged:
            for e in merged.get(key, []):
                e["merge_level"] = level
            return merged
    except Exception:  # noqa: BLE001
        pass
    # 回退：拼接所有编辑
    all_edits = []
    for p in patches:
        for e in get_payload_items(p, update_mode):
            e.setdefault("merge_level", level)
            all_edits.append(e)
    return {"reasoning": "fallback concatenation", payload_key(update_mode): all_edits}


def _hierarchical_merge(
    prompt_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    batch_size: int,
    verbose: bool,
    label: str = "",
    workers: int = 16,
    meta_prompt_context: str = "",
) -> dict:
    """使用给定 system prompt 层次化合并 N 个 patch。

    同层 batch 通过 ThreadPoolExecutor 并行执行。
    """
    if not patches:
        return {"reasoning": "no patches", payload_key(update_mode): []}
    if len(patches) == 1:
        return patches[0]

    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        batches: list[tuple[int, list[dict]]] = []
        for i in range(0, len(current), batch_size):
            batch = current[i : i + batch_size]
            batches.append((i, batch))

        if verbose:
            print(
                f"    [aggregate {label}] level={level}  "
                f"{len(current)} patches → {len(batches)} batches "
                f"(parallel, batch_size={batch_size})"
            )

        next_level: list[dict | None] = [None] * len(batches)

        to_merge: list[tuple[int, list[dict]]] = []
        for idx, (i, batch) in enumerate(batches):
            if len(batch) == 1:
                next_level[idx] = batch[0]
            else:
                to_merge.append((idx, batch))

        if to_merge:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _merge_batch, prompt_content, batch, system_prompt, update_mode,
                        meta_prompt_context, level,
                    ): idx
                    for idx, batch in to_merge
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    next_level[idx] = fut.result()
                    if verbose:
                        batch_i, batch_data = batches[idx]
                        n_edits = len(get_payload_items(next_level[idx], update_mode))
                        print(
                            f"      [aggregate {label}] level={level} "
                            f"batch [{batch_i}:{batch_i+len(batch_data)}] "
                            f"→ 1 patch ({n_edits} {payload_label(update_mode)})"
                        )

        current = [x for x in next_level if x is not None]

    return current[0]


# ── 公共 API ────────────────────────────────────────────────────────────────

def merge_patches(
    prompt_content: str,
    failure_patches: list[dict],
    success_patches: list[dict],
    batch_size: int = 8,
    verbose: bool = True,
    workers: int = 16,
    update_mode: str = "patch",
    meta_prompt_context: str = "",
) -> dict:
    """失败优先的层次化合并，并跟踪 support count。

    1. 独立合并失败 patch（并行）
    2. 独立合并成功 patch（并行）
    3. 最终合并：合并两组，失败组优先

    返回合并后的 :class:`~promptopt.types.Patch` dict（``edits`` + ``reasoning``）。
    """
    if verbose:
        print(
            f"    [3/6 AGGREGATE] "
            f"failure={len(failure_patches)} success={len(success_patches)} "
            f"(parallel, workers={workers})"
        )

    update_mode = normalize_update_mode(update_mode)
    if is_full_rewrite_minibatch_mode(update_mode):
        merge_failure_prompt = load_template("merge_failure_full_rewrite")
        merge_success_prompt = load_template("merge_success_full_rewrite")
        merge_final_prompt = load_template("merge_final_full_rewrite")
    elif is_rewrite_mode(update_mode):
        merge_failure_prompt = load_template("merge_failure_rewrite")
        merge_success_prompt = load_template("merge_success_rewrite")
        merge_final_prompt = load_template("merge_final_rewrite")
    else:
        merge_failure_prompt = load_template("merge_failure")
        merge_success_prompt = load_template("merge_success")
        merge_final_prompt = load_template("merge_final")

    failure_merged = _hierarchical_merge(
        prompt_content, failure_patches, merge_failure_prompt, update_mode,
        batch_size, verbose, label="failure", workers=workers,
        meta_prompt_context=meta_prompt_context,
    )

    success_merged = _hierarchical_merge(
        prompt_content, success_patches, merge_success_prompt, update_mode,
        batch_size, verbose, label="success", workers=workers,
        meta_prompt_context=meta_prompt_context,
    )

    f_edits = get_payload_items(failure_merged, update_mode)
    s_edits = get_payload_items(success_merged, update_mode)

    if not f_edits and not s_edits:
        return {"reasoning": "no updates from either group", payload_key(update_mode): []}
    if not s_edits:
        return failure_merged
    if not f_edits:
        return success_merged

    combined_patches = [failure_merged, success_merged]
    combined_text = json.dumps(combined_patches, ensure_ascii=False, indent=2)
    if is_full_rewrite_minibatch_mode(update_mode):
        item_label = payload_label(update_mode)
        user = (
            f"## Current Prompt\n{prompt_content}\n\n"
            f"## Two pre-merged candidate groups to combine\n"
            f"Group 1 (from failed trajectories): "
            f"{len(f_edits)} {item_label}\n"
            f"Group 2 (from successful trajectories): "
            f"{len(s_edits)} {item_label}\n\n"
            f"{combined_text}"
        )
    else:
        user = (
            f"## Current Prompt\n{prompt_content}\n\n"
            f"## Two pre-merged patch groups to combine\n"
            f"Group 1 (failure-driven, HIGH priority): "
            f"{len(f_edits)} edits\n"
            f"Group 2 (success-driven, lower priority): "
            f"{len(s_edits)} edits\n\n"
            f"{combined_text}"
        )
    optimizer_ctx = format_meta_prompt_context(meta_prompt_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=merge_final_prompt,
            user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(update_mode) else 4096,
            retries=3,
            stage="merge",
        )
        final = extract_json(response)
        key = payload_key(update_mode)
        if final and key in final:
            if verbose:
                print(
                    f"    [aggregate final] "
                    f"{len(f_edits)}+{len(s_edits)} → {len(final[key])} {payload_label(update_mode)}"
                )
            return final
    except Exception:  # noqa: BLE001
        pass

    return {
        "reasoning": "fallback: failure first, then success",
        payload_key(update_mode): f_edits + s_edits,
    }
