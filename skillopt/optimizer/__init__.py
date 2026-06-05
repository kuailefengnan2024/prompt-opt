"""【功能描述】SkillOpt 优化器 — skill 更新操作，类比神经网络训练中的 optimizer，将计算得到的「梯度」（patches）应用到当前 skill 文档以生成更新后的候选 skill。

【输入】各子模块的 apply_edit、apply_patch、rank_and_select 等 API 及对应参数。

【输出】导出 apply_edit、apply_patch、rank_and_select 等公共符号。

子模块
------
- skill: 编辑应用（optimizer.step() / 参数更新）
- clip: 编辑排序与选择（梯度裁剪）
- slow_update: 纵向比较与指导（EMA / 正则化）
- meta_skill: 跨 epoch 的优化器上下文记忆
"""
from skillopt.optimizer.skill import apply_edit, apply_patch  # noqa: F401
from skillopt.optimizer.clip import rank_and_select  # noqa: F401
