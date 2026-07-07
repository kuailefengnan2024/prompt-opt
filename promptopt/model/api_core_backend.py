# 【功能描述】api-core LLM 后端，供 Reflect optimizer 阶段调用
# 【输入】system/user 文本、llm_provider 环境配置
# 【输出】模型文本与用量 dict

from __future__ import annotations

import threading
import time
from typing import Any

from promptopt.clients.api_core_bridge import configure_api_clients, llm_chat_sync

_LLM_PROVIDER = "doubao21pro"
_TOKEN_LOCK = threading.Lock()
_TOKEN_TRACKER: dict[str, dict[str, int]] = {}


def configure_api_core_llm(*, llm_provider: str, image_provider: str) -> None:
    global _LLM_PROVIDER
    _LLM_PROVIDER = llm_provider
    configure_api_clients(llm_provider=llm_provider, image_provider=image_provider)


def _record(stage: str, usage: dict[str, Any]) -> dict[str, int]:
    with _TOKEN_LOCK:
        bucket = _TOKEN_TRACKER.setdefault(
            stage,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        bucket["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        bucket["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        return dict(bucket)


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    del max_completion_tokens, reasoning_effort, timeout
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err = ""
    for attempt in range(max(1, retries)):
        text, err = llm_chat_sync(messages, provider=_LLM_PROVIDER)
        if err:
            last_err = err
            time.sleep(min(2 ** attempt, 30))
            continue
        usage = _record(stage, {"prompt_tokens": len(system) // 4, "completion_tokens": len(text or "") // 4})
        return text or "", usage
    raise RuntimeError(f"api-core LLM 失败: {last_err}")


def get_token_summary() -> dict:
    with _TOKEN_LOCK:
        summary = {k: dict(v) for k, v in _TOKEN_TRACKER.items()}
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for stage, values in summary.items():
        total["calls"] += values["calls"]
        total["prompt_tokens"] += values["prompt_tokens"]
        total["completion_tokens"] += values["completion_tokens"]
        total["total_tokens"] += values["total_tokens"]
    summary["_total"] = total
    return summary


def reset_token_tracker() -> None:
    with _TOKEN_LOCK:
        _TOKEN_TRACKER.clear()
