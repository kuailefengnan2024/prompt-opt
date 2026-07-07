# 【功能描述】api-core 同步桥接：LLM 对话与文生图
# 【输入】provider 名、prompt/messages、image size
# 【输出】(result, error) 二元组；不向外抛未捕获异常

from __future__ import annotations

import asyncio
from typing import Any

_LLM_CLIENT: Any = None
_IMAGE_CLIENT: Any = None
_LLM_PROVIDER: str = ""
_IMAGE_PROVIDER: str = ""


def configure_api_clients(
    *,
    llm_provider: str,
    image_provider: str,
) -> None:
    global _LLM_CLIENT, _IMAGE_CLIENT, _LLM_PROVIDER, _IMAGE_PROVIDER
    from api_core import AIClient

    _LLM_PROVIDER = llm_provider
    _IMAGE_PROVIDER = image_provider
    _LLM_CLIENT = AIClient(llm_provider=llm_provider)
    _IMAGE_CLIENT = AIClient(image_provider=image_provider)


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def llm_chat_sync(
    messages: list[dict[str, str]],
    *,
    provider: str | None = None,
) -> tuple[str | None, str | None]:
    if _LLM_CLIENT is None:
        raise RuntimeError("api-core 未配置，请先调用 configure_api_clients()")
    client = _LLM_CLIENT
    if provider and provider != _LLM_PROVIDER:
        from api_core import AIClient

        client = AIClient(llm_provider=provider)
    return _run(client.llm.chat(messages))


def generate_image_sync(
    prompt: str,
    *,
    size: str,
    provider: str | None = None,
) -> tuple[bytes | None, str | None]:
    if _IMAGE_CLIENT is None:
        raise RuntimeError("api-core 未配置，请先调用 configure_api_clients()")
    client = _IMAGE_CLIENT
    if provider and provider != _IMAGE_PROVIDER:
        from api_core import AIClient

        client = AIClient(image_provider=provider)
    return _run(client.image.generate(prompt, size=size))
