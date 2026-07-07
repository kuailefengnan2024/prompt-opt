# 【功能描述】T2I Reflect 流水线核心类型
# 【输出】Edit、Patch、GateResult 等 dataclass

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Literal

from promptopt.evaluation.gate import GateAction, GateResult  # noqa: F401

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
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
    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        key = "edits" if "edits" in d else "rules"
        raw = d.get(key) or []
        return cls(
            edits=[Edit.from_dict(e) if isinstance(e, dict) else e for e in raw],
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


@dataclass
class FailureSummaryEntry:
    failure_type: str
    count: int
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> FailureSummaryEntry:
        return cls(
            failure_type=d.get("failure_type", ""),
            count=int(d.get("count", 0) or 0),
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
    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
    batch_size: int = 0
    failure_summary: list[FailureSummaryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> RawPatch:
        inner = d.get("patch", d)
        patch = Patch.from_dict(inner) if isinstance(inner, dict) else Patch()
        return cls(
            patch=patch,
            source_type=d.get("source_type", "failure"),
            batch_size=int(d.get("batch_size", 0) or 0),
            failure_summary=[
                FailureSummaryEntry.from_dict(fs)
                for fs in (d.get("failure_summary") or [])
                if isinstance(fs, dict)
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


@dataclass
class RolloutResult:
    id: str
    hard: int
    soft: float
    fail_reason: str = ""
    task_type: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        known = {f.name for f in dc_fields(cls)}
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")),
            hard=int(d.get("hard", 0) or 0),
            soft=float(d.get("soft", 0) or 0),
            fail_reason=str(d.get("fail_reason", "") or ""),
            task_type=str(d.get("task_type", "") or ""),
            extras=extras,
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "hard": self.hard,
            "soft": self.soft,
            "fail_reason": self.fail_reason,
            "task_type": self.task_type,
        }
        d.update(self.extras)
        return d
