"""【功能描述】ReflACT Evaluation：候选 prompt 校验与模型选择。
【输入】候选 prompt、selection 集 rollout 分数及当前/历史最优状态。
【输出】`GateResult` 接受/拒绝决策及 `evaluate_gate` 决策函数。
"""
from skillopt.evaluation.gate import evaluate_gate, GateAction, GateResult  # noqa: F401
