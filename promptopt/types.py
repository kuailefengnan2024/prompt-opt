"""【功能描述】Reflect 流水线标准化 I/O 类型：六阶段逐步流水线与两个 epoch 级阶段的共享 dataclass。
【输入】各阶段的 dict 表示（经 `from_dict` 解析）。
【输出】类型安全的 dataclass 及 `to_dict` 往返序列化；再导出 `GateResult`、`GateAction`、`BatchSpec`。

再导出
----------
GateResult, GateAction — 来自 promptopt.evaluation.gate
BatchSpec              — 来自 promptopt.datasets.base
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Literal

from promptopt.evaluation.gate import GateAction, GateResult  # noqa: F401
from promptopt.datasets.base import BatchSpec  # noqa: F401


# ── 原子类型 ─────────────────────────────────────────────────────────

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
    """对可优化 prompt 文档的单条编辑操作。

    用于 Reflect → Aggregate → Select → Update → MetaReflect 各阶段。
    """

    op: EditOp
    content: str = ""
    target: str = ""
    support_count: int | None = None
    source_type: Literal["failure", "success"] | None = None
    merge_level: int | None = None
    update_origin: str = ""
    update_target: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        return cls(
            op=d.get("op", "append"),
            content=d.get("content", ""),
            target=d.get("target", ""),
            support_count=d.get("support_count"),
            source_type=d.get("source_type"),
            merge_level=d.get("merge_level"),
            update_origin=d.get("update_origin", ""),
            update_target=d.get("update_target", ""),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target:
            d["target"] = self.target
        if self.support_count is not None:
            d["support_count"] = self.support_count
        if self.source_type is not None:
            d["source_type"] = self.source_type
        if self.merge_level is not None:
            d["merge_level"] = self.merge_level
        if self.update_origin:
            d["update_origin"] = self.update_origin
        if self.update_target:
            d["update_target"] = self.update_target
        return d


@dataclass
class Patch:
    """带推理说明的一组 edits。

    Aggregate（③）与 Select（④）的输出；Update（⑤）的输入。
    """

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        edits_raw = d.get("edits", [])
        return cls(
            edits=[Edit.from_dict(e) if isinstance(e, dict) else e for e in edits_raw],
            reasoning=d.get("reasoning", ""),
            ranking_details=d.get("ranking_details"),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "reasoning": self.reasoning,
            "edits": [e.to_dict() if isinstance(e, Edit) else e for e in self.edits],
        }
        if self.ranking_details is not None:
            d["ranking_details"] = self.ranking_details
        return d


# ── 阶段 ① ROLLOUT ──────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """单条 episode/任务 rollout 的结果。

    通用字段为必填；环境专有字段存放在 ``extras`` 中。
    """

    id: str
    hard: int
    soft: float
    n_turns: int = 0
    fail_reason: str = ""
    task_type: str = ""
    task_description: str = ""
    predicted_answer: str = ""
    question: str = ""
    reference_text: str = ""
    target_system_prompt: str = ""
    target_user_prompt: str = ""
    spreadsheet_preview: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False,  # type: ignore[assignment]
    )

    @classmethod
    def _get_known_fields(cls) -> frozenset[str]:
        if cls._KNOWN_FIELDS is None:
            cls._KNOWN_FIELDS = frozenset(
                f.name for f in dc_fields(cls)
                if f.name != "_KNOWN_FIELDS"
            )
        return cls._KNOWN_FIELDS

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        known = cls._get_known_fields()
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")),
            hard=int(d.get("hard", 0)),
            soft=float(d.get("soft", 0.0)),
            n_turns=int(d.get("n_turns", 0)),
            fail_reason=str(d.get("fail_reason", "")),
            task_type=str(d.get("task_type", "")),
            task_description=str(d.get("task_description", "")),
            predicted_answer=str(d.get("predicted_answer", "")),
            question=str(d.get("question", "")),
            reference_text=str(d.get("reference_text", "")),
            target_system_prompt=str(d.get("target_system_prompt", "")),
            target_user_prompt=str(d.get("target_user_prompt", "")),
            spreadsheet_preview=str(d.get("spreadsheet_preview", "")),
            extras=extras,
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "hard": self.hard,
            "soft": self.soft,
        }
        for attr in (
            "n_turns", "fail_reason", "task_type", "task_description",
            "predicted_answer", "question", "reference_text",
            "target_system_prompt", "target_user_prompt",
            "spreadsheet_preview",
        ):
            val = getattr(self, attr)
            if val:
                d[attr] = val
        d.update(self.extras)
        return d


# ── 阶段 ② REFLECT ──────────────────────────────────────────────────────

@dataclass
class FailureSummaryEntry:
    """错误分析师生成的失败摘要中的单条条目。"""

    failure_type: str
    count: int = 0
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> FailureSummaryEntry:
        return cls(
            failure_type=d.get("failure_type", ""),
            count=int(d.get("count", 0)),
            description=d.get("description", ""),
        )

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "description": self.description,
        }


@dataclass
class RawPatch:
    """Reflect 阶段的分析师输出 — 带来源信息的 patch。

    封装 ``run_error_analyst_minibatch`` 与
    ``run_success_analyst_minibatch`` 产生的 dict。
    """

    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
    batch_size: int = 0
    failure_summary: list[FailureSummaryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> RawPatch | None:
        if d is None:
            return None
        inner = d.get("patch", d)
        if not isinstance(inner, dict):
            return None
        patch = Patch.from_dict(inner)
        return cls(
            patch=patch,
            source_type=d.get("source_type", "failure"),
            batch_size=int(d.get("batch_size", 0)),
            failure_summary=[
                FailureSummaryEntry.from_dict(fs)
                for fs in d.get("failure_summary", [])
            ],
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "patch": self.patch.to_dict(),
            "source_type": self.source_type,
            "batch_size": self.batch_size,
        }
        if self.failure_summary:
            d["failure_summary"] = [fs.to_dict() for fs in self.failure_summary]
        return d


# ── Epoch 级：SLOW_UPDATE ─────────────────────────────────────────────

@dataclass
class SlowUpdateResult:
    """Epoch 级 slow update 阶段（EMA / 正则化）的输出。"""

    reasoning: str = ""
    slow_update_content: str = ""
    action: str = ""
    time_s: float | None = None
    prev_hard: float | None = None
    curr_hard: float | None = None
    selection_hard: float | None = None
    selection_soft: float | None = None
    candidate_hash: str = ""
    update_origin: str = ""
    update_target: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> SlowUpdateResult | None:
        if d is None:
            return None
        return cls(
            reasoning=d.get("reasoning", ""),
            slow_update_content=d.get("slow_update_content", ""),
            action=d.get("action", ""),
            time_s=d.get("time_s"),
            prev_hard=d.get("prev_hard"),
            curr_hard=d.get("curr_hard"),
            selection_hard=d.get("selection_hard"),
            selection_soft=d.get("selection_soft"),
            candidate_hash=d.get("candidate_hash", ""),
            update_origin=d.get("update_origin", ""),
            update_target=d.get("update_target", ""),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "reasoning": self.reasoning,
            "slow_update_content": self.slow_update_content,
        }
        if self.action:
            d["action"] = self.action
        if self.time_s is not None:
            d["time_s"] = self.time_s
        if self.prev_hard is not None:
            d["prev_hard"] = self.prev_hard
        if self.curr_hard is not None:
            d["curr_hard"] = self.curr_hard
        if self.selection_hard is not None:
            d["selection_hard"] = self.selection_hard
        if self.selection_soft is not None:
            d["selection_soft"] = self.selection_soft
        if self.candidate_hash:
            d["candidate_hash"] = self.candidate_hash
        if self.update_origin:
            d["update_origin"] = self.update_origin
        if self.update_target:
            d["update_target"] = self.update_target
        return d
