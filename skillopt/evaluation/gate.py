"""【功能描述】验证门（validation gate）：接受或拒绝候选 skill，类比 NN 训练中的验证早停与模型选择。
【输入】候选 skill、候选 hard 分、当前/历史最优 skill 与分数及 `global_step`。
【输出】不可变 `GateResult`（`accept_new_best` / `accept` / `reject` 及更新后的状态字段）。

比较候选分数与当前分、历史最优分后返回纯决策结果；trainer 负责副作用（缓存、rollout、打印、状态变更）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass(frozen=True)
class GateResult:
    """验证门决策的不可变结果。"""

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


def evaluate_gate(
    candidate_skill: str,
    cand_hard: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    best_step: int,
    global_step: int,
) -> GateResult:
    """纯门控决策：将候选分数与 current/best 比较。

    返回带更新状态的 *GateResult*；由调用方决定后续动作（打印、变更 trainer 状态、日志等）。
    """
    if cand_hard > current_score:
        new_current_skill = candidate_skill
        new_current_score = cand_hard
        if cand_hard > best_score:
            return GateResult(
                action="accept_new_best",
                current_skill=new_current_skill,
                current_score=new_current_score,
                best_skill=candidate_skill,
                best_score=cand_hard,
                best_step=global_step,
            )
        return GateResult(
            action="accept",
            current_skill=new_current_skill,
            current_score=new_current_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
        )
    return GateResult(
        action="reject",
        current_skill=current_skill,
        current_score=current_score,
        best_skill=best_skill,
        best_score=best_score,
        best_step=best_step,
    )
