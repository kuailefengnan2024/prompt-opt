"""【功能描述】ReflACT（Reflective Agent Tuning）：通过结构化反思与自我改进迭代优化 LLM Agent 技能的通用框架。
【输入】无（包级元数据与类型再导出）。
【输出】`__version__` 及流水线核心类型（BatchSpec、Edit、Patch、RolloutResult 等）。

流水线阶段：
  1. Rollout   — 使用当前 skill 执行 episode
  2. Reflect   — 分析轨迹并生成 patch
  3. Aggregate — 分层合并 patch
  4. Select    — 排序并选取 top edits
  5. Update    — 将 edits 应用到 skill 文档
  6. Evaluate  — 校验候选 skill，接受/拒绝
"""

__version__ = "0.1.0"

from skillopt.types import (  # noqa: F401
    BatchSpec,
    Edit,
    EditOp,
    FailureSummaryEntry,
    GateAction,
    GateResult,
    Patch,
    RawPatch,
    RolloutResult,
    SlowUpdateResult,
)
