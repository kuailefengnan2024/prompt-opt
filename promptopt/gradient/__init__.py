"""【功能描述】prompt-opt 梯度 — trajectory 分析与 patch 生成，类比神经网络训练中的梯度计算：分析 minibatch rollout trajectory 以产生 prompt 编辑 patch（驱动 prompt 更新的「梯度」）。

【输入】各子模块的 run_minibatch_reflect、merge_patches 等 API 及对应参数。

【输出】导出 run_minibatch_reflect、merge_patches 等公共符号。

子模块
-------
- reflect: minibatch trajectory 分析（梯度计算）
- aggregate: 层次化 patch 合并（梯度聚合）
"""
from promptopt.gradient.reflect import (  # noqa: F401
    run_minibatch_reflect,
)
from promptopt.gradient.aggregate import merge_patches  # noqa: F401
