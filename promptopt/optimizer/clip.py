"""【功能描述】Reflect 梯度裁剪 — 由 LLM 驱动的编辑排序与选择，类比神经网络训练中的梯度裁剪：按重要性排序候选编辑并选取 top-L 应用，控制有效步长。

【输入】prompt_content、patch、max_edits、meta_prompt_context、update_mode。

【输出】含选中编辑及可选 ranking_details 的 Patch dict；原 core/select.py。
"""
from __future__ import annotations

from promptopt.model import chat_optimizer
from promptopt.optimizer.meta_prompt import format_meta_prompt_context
from promptopt.optimizer.update_modes import (
    describe_item,
    get_payload_items,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from promptopt.templates import fill_prompt, has_prompt
from promptopt.utils import extract_json


# ── 公共 API ────────────────────────────────────────────────────────────────

def rank_and_select(
    prompt_content: str,
    patch: dict,
    max_edits: int,
    meta_prompt_context: str = "",
    update_mode: str = "patch",
) -> dict:
    """使用 optimizer LLM 按重要性排序编辑，保留 top-L。

    若编辑池在预算内，原样返回 patch；否则调用 optimizer 排序并选取影响最大的编辑。

    Parameters
    ----------
    prompt_content : str
        当前 prompt 文档。
    patch : dict
        含 ``edits`` 列表的合并 :class:`~promptopt.types.Patch` dict。
    max_edits : int
        保留的最大编辑数（「编辑预算」）。

    Returns
    -------
    dict
        含选中编辑及可选 ``ranking_details`` 的 :class:`~promptopt.types.Patch` dict。
    """
    update_mode = normalize_update_mode(update_mode)
    edits = get_payload_items(patch, update_mode)
    if len(edits) <= max_edits:
        return patch

    # 为 optimizer 构建编辑池描述
    edits_desc = []
    for i, edit in enumerate(edits):
        edits_desc.append(f"[{i}] {describe_item(edit, update_mode, max_chars=500)}")
    edits_pool = "\n".join(edits_desc)

    optimizer_ctx = format_meta_prompt_context(meta_prompt_context)
    if not has_prompt("ranking"):
        raise FileNotFoundError("缺少模板 promptopt/prompts/ranking.md")
    user = fill_prompt("ranking", {
        "current_prompt": prompt_content,
        "payload_label": payload_label(update_mode),
        "edit_count": str(len(edits)),
        "edit_budget": str(max_edits),
        "edits_pool": edits_pool,
    })
    if optimizer_ctx.strip():
        user = f"{optimizer_ctx.strip()}\n\n{user}"

    try:
        response, _ = chat_optimizer(
            system="", user=user,
            max_completion_tokens=2048, retries=3, stage="ranking",
        )
        result = extract_json(response)
        if result and "selected_indices" in result:
            indices = result["selected_indices"]
            selected = []
            seen: set[int] = set()
            for idx in indices:
                if (
                    isinstance(idx, int)
                    and 0 <= idx < len(edits)
                    and idx not in seen
                ):
                    selected.append(edits[idx])
                    seen.add(idx)
                if len(selected) >= max_edits:
                    break
            if selected:
                return {
                    "reasoning": patch.get("reasoning", "")
                    + f" [optimizer-ranked: selected {len(selected)}/{len(edits)} {payload_label(update_mode)}]",
                    payload_key(update_mode): selected,
                    "ranking_details": result,
                }
    except Exception:  # noqa: BLE001
        pass

    # 回退：简单截断
    return {
        "reasoning": patch.get("reasoning", "")
        + f" [fallback truncated {len(edits)}->{max_edits} {payload_label(update_mode)}]",
        payload_key(update_mode): edits[:max_edits],
    }
