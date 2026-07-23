"""【功能描述】从 episode 结果计算 hard/soft 准确率。
【输入】episode 结果列表（dict 或带 hard/soft 属性的对象）。
【输出】`(hard, soft)` 元组。
"""
from __future__ import annotations


def compute_score(results: list) -> tuple[float, float]:
    """从 episode 结果列表计算 hard 与 soft 准确率。"""
    if not results:
        return 0.0, 0.0

    def _hard(r: object) -> int:
        return int(r.hard if hasattr(r, "hard") else r.get("hard", 0))  # type: ignore[union-attr]

    def _soft(r: object) -> float:
        return float(r.soft if hasattr(r, "soft") else r.get("soft", 0.0))  # type: ignore[union-attr]

    hard = sum(_hard(r) for r in results) / len(results)
    soft = sum(_soft(r) for r in results) / len(results)
    return hard, soft
