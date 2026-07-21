"""【功能描述】Reflect Aggregate 阶段 — 层次化 patch 合并；将 Reflect 阶段独立生成的 patch 通过层次化 LLM 调用合并为单一连贯 patch，失败驱动 patch 优先于成功驱动 patch。

【输入】prompt_content、failure_patches、success_patches、batch_size、update_mode 等。

【输出】合并后的 Patch dict（含 edits/reasoning 或对应 payload 键）。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from promptopt.model import chat_optimizer
from promptopt.design_requirement import format_design_requirement_section
from promptopt.optimizer.meta_prompt import format_meta_prompt_context
from promptopt.optimizer.update_modes import (
    get_payload_items,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from promptopt.optimizer.clip import clip_edits
from promptopt.templates import fill_prompt, get_prompt_scope_section, has_prompt
from promptopt.utils import extract_json


# ── 内部辅助函数 ──────────────────────────────────────────────────────────

def _merge_batch(
    prompt_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    meta_prompt_context: str = "",
    design_requirement: str = "",
    edit_budget: int = 4,
    level: int = 1,
    label: str = "",
) -> dict:
    """调用 optimizer LLM 将一批 patch 合并为一个。"""
    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    optimizer_ctx = format_meta_prompt_context(meta_prompt_context)
    design_ctx = format_design_requirement_section(design_requirement)
    scope_ctx = get_prompt_scope_section()

    if not has_prompt("merge_failure"):
        raise FileNotFoundError("缺少模板 promptopt/prompts/merge_failure.md")
    tmpl = "merge_failure" if label == "failure" else "merge_success"
    if not has_prompt(tmpl):
        raise FileNotFoundError(f"缺少模板 promptopt/prompts/{tmpl}.md")
    user = fill_prompt(tmpl, {
        "current_prompt": prompt_content,
        "patch_count": str(len(patches)),
        "merge_level": str(level),
        "patches_json": patches_text,
        "design_requirement_section": design_ctx.strip(),
        "prompt_scope_section": scope_ctx,
        "edit_budget": str(edit_budget),
    })
    prefix_parts = [p for p in (scope_ctx, design_ctx.strip(), optimizer_ctx.strip()) if p]
    if prefix_parts:
        user = "\n\n".join(prefix_parts) + "\n\n" + user
    system = ""
    try:
        response, _ = chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=4096,
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
    design_requirement: str = "",
    edit_budget: int = 4,
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
                        meta_prompt_context, design_requirement, edit_budget, level, label,
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


def _finalize_merged_patch(
    patch: dict,
    *,
    edit_budget: int,
    update_mode: str,
    verbose: bool,
) -> dict:
    clipped = clip_edits(patch, max_edits=edit_budget, update_mode=update_mode)
    details = clipped.get("clip_details")
    if verbose and isinstance(details, dict) and details.get("input_count", 0) > details.get("output_count", 0):
        print(
            f"    [3/5 CLIP] rule {details['input_count']}→{details['output_count']} "
            f"{payload_label(update_mode)} (budget={edit_budget})"
        )
    return clipped


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
    design_requirement: str = "",
    edit_budget: int = 4,
) -> dict:
    """失败优先的层次化合并，并跟踪 support count。

    1. 独立合并失败 patch（并行）
    2. 独立合并成功 patch（并行）
    3. 最终合并：合并两组，失败组优先

    返回合并后的 :class:`~promptopt.types.Patch` dict（``edits`` + ``reasoning``）。
    """
    if verbose:
        print(
            f"    [3/5 AGGREGATE] "
            f"failure={len(failure_patches)} success={len(success_patches)} "
            f"(parallel, workers={workers})"
        )

    update_mode = normalize_update_mode(update_mode)

    failure_merged = _hierarchical_merge(
        prompt_content, failure_patches, "", update_mode,
        batch_size, verbose, label="failure", workers=workers,
        meta_prompt_context=meta_prompt_context,
        design_requirement=design_requirement,
        edit_budget=edit_budget,
    )

    success_merged = _hierarchical_merge(
        prompt_content, success_patches, "", update_mode,
        batch_size, verbose, label="success", workers=workers,
        meta_prompt_context=meta_prompt_context,
        design_requirement=design_requirement,
        edit_budget=edit_budget,
    )

    f_edits = get_payload_items(failure_merged, update_mode)
    s_edits = get_payload_items(success_merged, update_mode)

    if not f_edits and not s_edits:
        return _finalize_merged_patch(
            {"reasoning": "no updates from either group", payload_key(update_mode): []},
            edit_budget=edit_budget,
            update_mode=update_mode,
            verbose=verbose,
        )
    if not s_edits:
        return _finalize_merged_patch(
            failure_merged,
            edit_budget=edit_budget,
            update_mode=update_mode,
            verbose=verbose,
        )
    if not f_edits:
        return _finalize_merged_patch(
            success_merged,
            edit_budget=edit_budget,
            update_mode=update_mode,
            verbose=verbose,
        )

    combined_patches = [failure_merged, success_merged]
    combined_text = json.dumps(combined_patches, ensure_ascii=False, indent=2)
    optimizer_ctx = format_meta_prompt_context(meta_prompt_context)
    design_ctx = format_design_requirement_section(design_requirement)
    scope_ctx = get_prompt_scope_section()
    if not has_prompt("merge_final"):
        raise FileNotFoundError("缺少模板 promptopt/prompts/merge_final.md")
    user = fill_prompt("merge_final", {
        "current_prompt": prompt_content,
        "failure_edit_count": str(len(f_edits)),
        "success_edit_count": str(len(s_edits)),
        "combined_patches_json": combined_text,
        "design_requirement_section": design_ctx.strip(),
        "prompt_scope_section": scope_ctx,
        "edit_budget": str(edit_budget),
    })
    prefix_parts = [p for p in (scope_ctx, design_ctx.strip(), optimizer_ctx.strip()) if p]
    if prefix_parts:
        user = "\n\n".join(prefix_parts) + "\n\n" + user
    try:
        response, _ = chat_optimizer(
            system="",
            user=user,
            max_completion_tokens=4096,
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
            return _finalize_merged_patch(
                final,
                edit_budget=edit_budget,
                update_mode=update_mode,
                verbose=verbose,
            )
    except Exception:  # noqa: BLE001
        pass

    return _finalize_merged_patch(
        {
            "reasoning": "fallback: failure first, then success",
            payload_key(update_mode): f_edits + s_edits,
        },
        edit_budget=edit_budget,
        update_mode=update_mode,
        verbose=verbose,
    )
