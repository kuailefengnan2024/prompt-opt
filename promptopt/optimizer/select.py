"""【功能描述】向后兼容存根 — rank_and_select 已迁移至 promptopt.optimizer.clip。

【输入】无（重导出 clip 模块符号）。

【输出】导出 rank_and_select。
"""
from promptopt.optimizer.clip import rank_and_select  # noqa: F401

__all__ = ["rank_and_select"]
