# 【功能描述】优化器 LLM 后端配置（T2I 固定 api_core）
# 【输入】set_optimizer_backend 调用
# 【输出】OPTIMIZER_BACKEND 全局状态

from __future__ import annotations

import os

OPTIMIZER_BACKEND = "api_core"


def set_optimizer_backend(backend: str) -> None:
    global OPTIMIZER_BACKEND
    normalized = str(backend or "api_core").strip().lower()
    if normalized != "api_core":
        raise ValueError(f"T2I 引擎仅支持 api_core，收到: {backend!r}")
    OPTIMIZER_BACKEND = normalized
    os.environ["OPTIMIZER_BACKEND"] = OPTIMIZER_BACKEND


def get_optimizer_backend() -> str:
    return OPTIMIZER_BACKEND
