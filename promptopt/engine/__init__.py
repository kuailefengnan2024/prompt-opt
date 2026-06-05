"""【功能描述】Reflect Engine：训练运行器，类比 mmengine 的 Runner，编排完整训练流水线。
【输入】扁平化配置 `cfg`、环境 `EnvAdapter`、数据加载器等。
【输出】`ReflectTrainer` 类（rollout、梯度、聚合、优化、评估各阶段）。

包含 rollout、梯度计算、聚合、优化与评估等阶段的编排逻辑。
"""
from promptopt.engine.trainer import ReflectTrainer  # noqa: F401

__all__ = ["ReflectTrainer"]
