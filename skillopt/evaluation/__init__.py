"""【功能描述】ReflACT Evaluation：候选 skill 校验与模型选择，类比验证集早停与模型选择。
【输入】候选 skill、selection 集 rollout 分数及当前/历史最优状态。
【输出】`GateResult` 接受/拒绝决策及 `evaluate_gate` 决策函数。

在 held-out selection 集上评估候选 skill，决定是否接受提议的更新。
"""
from skillopt.evaluation.gate import evaluate_gate, GateAction, GateResult  # noqa: F401
