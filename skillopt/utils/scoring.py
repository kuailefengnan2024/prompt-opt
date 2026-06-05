"""【功能描述】打分与哈希工具：从 episode 结果计算 hard/soft 准确率，并对 skill 内容生成短哈希。
【输入】episode 结果列表（dict 或 `RolloutResult`）；skill 文档字符串。
【输出】`(hard, soft)` 元组；用于缓存的 16 位 `skill_hash` 十六进制串。
"""
from __future__ import annotations

import hashlib


def compute_score(results: list) -> tuple[float, float]:
    """从 episode 结果列表计算 hard 与 soft 准确率。

    同时接受普通 dict 与 :class:`~skillopt.types.RolloutResult` 实例。
    """
    if not results:
        return 0.0, 0.0

    def _hard(r: object) -> int:
        return int(r.hard if hasattr(r, "hard") else r.get("hard", 0))  # type: ignore[union-attr]

    def _soft(r: object) -> float:
        return float(r.soft if hasattr(r, "soft") else r.get("soft", 0.0))  # type: ignore[union-attr]

    hard = sum(_hard(r) for r in results) / len(results)
    soft = sum(_soft(r) for r in results) / len(results)
    return hard, soft


def skill_hash(content: str) -> str:
    """返回 skill 内容的短确定性哈希（用于缓存）。"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
