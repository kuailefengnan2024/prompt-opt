"""【功能描述】prompt-opt 优化器 — 可优化 prompt 文档更新，类比 NN optimizer，将 patches 应用到当前 prompt 生成候选。

【输入】各子模块的 apply_edit、apply_patch、rank_and_select 等 API 及对应参数。

【输出】导出 apply_edit、apply_patch、rank_and_select 等公共符号。

子模块
------
- prompt_editor: 编辑应用（optimizer.step() / 参数更新）
- clip: 编辑排序与选择（梯度裁剪）
- slow_update: 纵向比较与指导（EMA / 正则化）
- meta_prompt: 跨 epoch 的优化器上下文记忆
"""
from promptopt.optimizer.prompt_editor import apply_edit, apply_patch  # noqa: F401
from promptopt.optimizer.clip import rank_and_select  # noqa: F401
