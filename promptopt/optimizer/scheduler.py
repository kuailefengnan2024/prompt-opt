"""【功能描述】Reflect 的学习率（编辑预算）调度器；Reflect 中的「学习率」指每步优化允许的最大 prompt 编辑数，调度器控制该预算在训练过程中的变化。

【输入】mode、max_lr、min_lr、total_steps 等调度配置参数。

【输出】LRScheduler 实例及其 step()/get_lr() 返回的每步编辑预算。

支持的模式
----------
- ``constant``   : 训练全程固定预算。
- ``linear``     : 从 ``max_lr`` 线性衰减至 ``min_lr``。
- ``cosine``     : 从 ``max_lr`` 余弦退火至 ``min_lr``。
- ``autonomous`` : 无上限 — 由模型自行决定编辑数量。

用法::

    scheduler = build_scheduler(cfg)
    for step in range(1, total_steps + 1):
        lr = scheduler.step()       # 返回本步编辑预算
        # ... 将 lr 用作 max_edits ...
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """编辑预算调度器基类。"""

    def __init__(self, max_lr: int, min_lr: int, total_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self._current_step = 0

    @abstractmethod
    def _compute_lr(self, step: int) -> int:
        """返回给定 1-indexed step 的编辑预算。"""

    def step(self) -> int:
        """前进一步并返回编辑预算。"""
        self._current_step += 1
        return self._compute_lr(self._current_step)

    def get_lr(self, step: int) -> int:
        """返回任意 step（1-indexed）的编辑预算。"""
        return self._compute_lr(step)

    def state_dict(self) -> dict:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: dict) -> None:
        self._current_step = state.get("current_step", 0)


class ConstantScheduler(LRScheduler):
    """训练全程固定编辑预算。"""

    def _compute_lr(self, step: int) -> int:
        return self.max_lr


class LinearScheduler(LRScheduler):
    """在 ``total_steps`` 内从 ``max_lr`` 线性衰减至 ``min_lr``。"""

    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.max_lr + (self.min_lr - self.max_lr) * t
        return max(self.min_lr, round(lr))


class CosineScheduler(LRScheduler):
    """在 ``total_steps`` 内从 ``max_lr`` 余弦退火至 ``min_lr``。"""

    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        return max(self.min_lr, round(lr))


class AutonomousScheduler(LRScheduler):
    """无编辑上限 — 由模型自由决定。"""

    NO_LIMIT = 999

    def _compute_lr(self, step: int) -> int:
        return self.NO_LIMIT


# ── 工厂 ──────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[LRScheduler]] = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "autonomous": AutonomousScheduler,
}


def build_scheduler(
    mode: str = "constant",
    max_lr: int = 8,
    min_lr: int = 2,
    total_steps: int = 8,
) -> LRScheduler:
    """根据配置参数构建调度器。

    Parameters
    ----------
    mode : str
        可选 ``constant``、``linear``、``cosine``、``autonomous`` 之一。
    max_lr : int
        初始 / 最大编辑预算。
    min_lr : int
        最小编辑预算（用于衰减模式）。
    total_steps : int
        训练中的优化步总数。
    """
    if mode not in _REGISTRY:
        raise ValueError(
            f"Unknown scheduler mode '{mode}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[mode](max_lr=max_lr, min_lr=min_lr, total_steps=total_steps)
