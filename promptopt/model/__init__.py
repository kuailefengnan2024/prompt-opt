# 【功能描述】T2I 引擎 LLM 调用入口（api-core 单后端）
# 【输入】chat_optimizer 的 system/user 消息
# 【输出】模型文本与 token 汇总

from __future__ import annotations

from promptopt.model import api_core_backend as _api_core
from promptopt.model.backend_config import (  # noqa: F401
    get_optimizer_backend,
    set_optimizer_backend,
)

__all__ = [
    "chat_optimizer",
    "get_token_summary",
    "reset_token_tracker",
    "get_optimizer_backend",
    "set_optimizer_backend",
]


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    del reasoning_effort
    return _api_core.chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
    )


def get_token_summary() -> dict:
    return _api_core.get_token_summary()


def reset_token_tracker() -> None:
    _api_core.reset_token_tracker()
