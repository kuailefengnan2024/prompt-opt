"""【功能描述】Reflect prompt 操作 — 编辑应用与 patch 处理。硬保护约束层（【核心特征】【画面风格】），仅允许改动可改层（【画面构图】：元素及其构图状态、布局相关色彩）。

【输入】prompt 文档字符串、Edit/Patch 实例或 dict。

【输出】apply_edit/apply_patch 返回更新后的 prompt；apply_patch_with_report 另返回逐编辑报告。
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptopt.types import Edit as EditType, Patch as PatchType

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

_LOCKED_TITLES = ("【核心特征】", "【画面风格】")
_COMPOSITION_TITLE = "【画面构图】"
_SECTION_START_RE = re.compile(r"【[^】]+】")
_MAX_REPLACE_RATIO = 0.35  # 无段落标题时防止误改整段约束/风格


def _replace_target_too_large(prompt: str, target: str) -> bool:
    p = (prompt or "").strip()
    t = (target or "").strip()
    if not p or not t:
        return False
    return len(t) > len(p) * _MAX_REPLACE_RATIO


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


def _section_span(prompt: str, title: str) -> tuple[int, int] | None:
    """返回某段标题起至下一【…】标题前（或文末）的 [start, end)。"""
    start = prompt.find(title)
    if start == -1:
        return None
    after = start + len(title)
    nxt = _SECTION_START_RE.search(prompt, after)
    end = nxt.start() if nxt else len(prompt)
    return start, end


def _extract_section(prompt: str, title: str) -> str | None:
    span = _section_span(prompt, title)
    if span is None:
        return None
    return prompt[span[0]:span[1]]


def _locked_sections_snapshot(prompt: str) -> dict[str, str]:
    snap: dict[str, str] = {}
    for title in _LOCKED_TITLES:
        body = _extract_section(prompt, title)
        if body is not None:
            snap[title] = body
    return snap


def _locked_sections_intact(before: str, after: str) -> bool:
    """若原文含锁定段，则要求这些段在 after 中完全一致。"""
    for title, body in _locked_sections_snapshot(before).items():
        if _extract_section(after, title) != body:
            return False
    return True


def _target_touches_locked(prompt: str, target: str) -> bool:
    """target 落在【核心特征】或【画面风格】内则 True。"""
    if not target:
        return False
    idx = prompt.find(target)
    if idx == -1:
        return False
    for title in _LOCKED_TITLES:
        span = _section_span(prompt, title)
        if span and span[0] <= idx < span[1]:
            return True
    return False


def _has_composition_section(prompt: str) -> bool:
    return _COMPOSITION_TITLE in prompt


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

    if target and _target_touches_locked(prompt, target):
        report["status"] = "skipped_locked_section"
        return prompt, report

    # 有三段结构时禁止整篇 append（会落到风格段后）；改为插入构图段末尾
    if op == "append" and _has_composition_section(prompt):
        span = _section_span(prompt, _COMPOSITION_TITLE)
        if span is None:
            report["status"] = "skipped_append_no_composition"
            return prompt, report
        _, end = span
        insert_at = end
        # 尽量插在构图段末尾空白之前
        body = prompt[span[0]:end].rstrip()
        insert_at = span[0] + len(body)
        candidate = prompt[:insert_at] + "\n\n" + content + "\n" + prompt[insert_at:]
        if not _locked_sections_intact(prompt, candidate):
            report["status"] = "skipped_would_mutate_locked_section"
            return prompt, report
        report["status"] = "applied_append_in_composition"
        return candidate, report

    if op == "append":
        su_start = prompt.find(SLOW_UPDATE_START)
        if su_start != -1:
            before = prompt[:su_start].rstrip()
            after = prompt[su_start:]
            candidate = before + "\n\n" + content + "\n\n" + after
            report["status"] = "applied_append_before_slow_update"
            return candidate, report
        report["status"] = "applied_append"
        return prompt.rstrip() + "\n\n" + content + "\n", report

    if op == "insert_after":
        if not target or target not in prompt:
            if _has_composition_section(prompt):
                report["status"] = "skipped_insert_after_target_not_found"
                return prompt, report
            su_start = prompt.find(SLOW_UPDATE_START)
            if su_start != -1:
                before = prompt[:su_start].rstrip()
                after = prompt[su_start:]
                report["status"] = "applied_insert_after_fallback_before_slow_update"
                return before + "\n\n" + content + "\n\n" + after, report
            report["status"] = "applied_insert_after_fallback_append"
            return prompt.rstrip() + "\n\n" + content + "\n", report
        if _target_touches_locked(prompt, target) and target != _COMPOSITION_TITLE:
            report["status"] = "skipped_locked_section"
            return prompt, report
        idx = prompt.index(target) + len(target)
        newline = prompt.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(prompt)
        candidate = prompt[:insert_at] + "\n" + content + "\n" + prompt[insert_at:]
        if not _locked_sections_intact(prompt, candidate):
            report["status"] = "skipped_would_mutate_locked_section"
            return prompt, report
        report["status"] = "applied_insert_after"
        return candidate, report

    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return prompt, report
        if target not in prompt:
            report["status"] = "skipped_replace_target_not_found"
            return prompt, report
        if _target_touches_locked(prompt, target):
            report["status"] = "skipped_locked_section"
            return prompt, report
        if _replace_target_too_large(prompt, target):
            report["status"] = "skipped_replace_target_too_large"
            return prompt, report
        candidate = prompt.replace(target, content, 1)
        if not _locked_sections_intact(prompt, candidate):
            report["status"] = "skipped_would_mutate_locked_section"
            return prompt, report
        report["status"] = "applied_replace"
        return candidate, report

    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return prompt, report
        if target not in prompt:
            report["status"] = "skipped_delete_target_not_found"
            return prompt, report
        if _target_touches_locked(prompt, target):
            report["status"] = "skipped_locked_section"
            return prompt, report
        candidate = prompt.replace(target, "", 1)
        if not _locked_sections_intact(prompt, candidate):
            report["status"] = "skipped_would_mutate_locked_section"
            return prompt, report
        report["status"] = "applied_delete"
        return candidate, report

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

    针对受保护 slow-update 区域及锁定段的编辑会被静默跳过。
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
