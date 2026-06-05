"""【功能描述】Reflect prompt 操作 — 编辑应用与 patch 处理，Reflect 流水线 Update 阶段（⑤）：将排序后的编辑集应用到当前 prompt 文档，生成更新候选，类比神经网络训练中的 optimizer.step()。

【输入】prompt 文档字符串、Edit/Patch 实例或 dict。

【输出】apply_edit/apply_patch 返回更新后的 prompt；apply_patch_with_report 另返回逐编辑报告。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptopt.types import Edit as EditType, Patch as PatchType

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"


def _is_in_slow_update_region(prompt: str, target: str) -> bool:
    """检查 *target* 文本是否落在受保护的 slow update 区域内。"""
    start_idx = prompt.find(SLOW_UPDATE_START)
    end_idx = prompt.find(SLOW_UPDATE_END)
    if start_idx == -1 or end_idx == -1:
        return False
    target_idx = prompt.find(target)
    if target_idx == -1:
        return False
    region_end = end_idx + len(SLOW_UPDATE_END)
    return start_idx <= target_idx < region_end


def _strip_slow_update_markers(text: str) -> str:
    """从编辑内容中移除 SLOW_UPDATE 标记，防止重复。"""
    return (
        text.replace(SLOW_UPDATE_START, "")
            .replace(SLOW_UPDATE_END, "")
    )


def _edit_fields(edit: EditType | dict) -> tuple[str, str, str]:
    op = edit.op if hasattr(edit, "op") else edit.get("op", "")
    content = _strip_slow_update_markers(
        (edit.content if hasattr(edit, "content") else edit.get("content", "")).strip()
    )
    target = edit.target if hasattr(edit, "target") else edit.get("target", "")
    return op, content, target


def _apply_edit_with_report(prompt: str, edit: EditType | dict) -> tuple[str, dict]:
    op, content, target = _edit_fields(edit)
    report = {
        "op": op,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
    }

    if target and _is_in_slow_update_region(prompt, target):
        report["status"] = "skipped_protected_slow_update_region"
        return prompt, report

    if op == "append":
        su_start = prompt.find(SLOW_UPDATE_START)
        if su_start != -1:
            before = prompt[:su_start].rstrip()
            after = prompt[su_start:]
            report["status"] = "applied_append_before_slow_update"
            return before + "\n\n" + content + "\n\n" + after, report
        report["status"] = "applied_append"
        return prompt.rstrip() + "\n\n" + content + "\n", report

    if op == "insert_after":
        if not target or target not in prompt:
            su_start = prompt.find(SLOW_UPDATE_START)
            if su_start != -1:
                before = prompt[:su_start].rstrip()
                after = prompt[su_start:]
                report["status"] = "applied_insert_after_fallback_before_slow_update"
                return before + "\n\n" + content + "\n\n" + after, report
            report["status"] = "applied_insert_after_fallback_append"
            return prompt.rstrip() + "\n\n" + content + "\n", report
        idx = prompt.index(target) + len(target)
        newline = prompt.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(prompt)
        report["status"] = "applied_insert_after"
        return prompt[:insert_at] + "\n" + content + "\n" + prompt[insert_at:], report

    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return prompt, report
        if target not in prompt:
            report["status"] = "skipped_replace_target_not_found"
            return prompt, report
        report["status"] = "applied_replace"
        return prompt.replace(target, content, 1), report

    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return prompt, report
        if target not in prompt:
            report["status"] = "skipped_delete_target_not_found"
            return prompt, report
        report["status"] = "applied_delete"
        return prompt.replace(target, "", 1), report

    report["status"] = "skipped_unknown_op"
    return prompt, report


def apply_edit(prompt: str, edit: EditType | dict) -> str:
    """将单条编辑操作应用到 prompt 文档。

    Parameters
    ----------
    prompt : str
        当前 prompt 文档内容。
    edit : Edit | dict
        :class:`~promptopt.types.Edit` 实例或含 ``op``、``content``、``target`` 键的普通 dict。

    针对受保护 slow-update 区域的编辑会被静默跳过。
    """
    updated_prompt, _ = _apply_edit_with_report(prompt, edit)
    return updated_prompt


def apply_patch_with_report(
    prompt: str,
    patch: PatchType | dict,
) -> tuple[str, list[dict]]:
    """应用 patch 并返回逐编辑报告，便于观测。"""
    edits = patch.edits if hasattr(patch, "edits") else patch.get("edits", [])
    reports: list[dict] = []
    for idx, edit in enumerate(edits, 1):
        try:
            prompt, report = _apply_edit_with_report(prompt, edit)
            report["index"] = idx
        except Exception as exc:  # noqa: BLE001
            report = {
                "index": idx,
                "op": "",
                "target": "",
                "content_preview": "",
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)
    return prompt, reports


def apply_patch(prompt: str, patch: PatchType | dict) -> str:
    """顺序将 patch（编辑列表）应用到 prompt 文档。

    Parameters
    ----------
    prompt : str
        当前 prompt 文档内容。
    patch : Patch | dict
        :class:`~promptopt.types.Patch` 实例或含 ``edits`` 编辑操作列表键的普通 dict。
    """
    updated_prompt, _ = apply_patch_with_report(prompt, patch)
    return updated_prompt
