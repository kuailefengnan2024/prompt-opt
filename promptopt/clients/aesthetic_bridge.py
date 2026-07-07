# 【功能描述】aesthetic-core 单图审美打分桥接（单系 ensemble）
# 【输入】图片 bytes、design_requirement、category、provider 映射
# 【输出】ComprehensiveResult dict；同步 comprehensive 调用

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

_AESTHETIC_CLIENT: Any = None
_STORAGE_ROOT: Path | None = None


def configure_aesthetic_storage(root: str | Path) -> None:
    global _STORAGE_ROOT
    _STORAGE_ROOT = Path(root)
    _STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


def build_aesthetic_client(
    ensemble_providers: dict[str, str],
) -> Any:
    global _AESTHETIC_CLIENT
    from api_core import AIClient
    from aesthetic_core import AestheticClient
    from aesthetic_core.evaluate.evaluate_constants import (
        apply_single_config,
        vision_limits_for_provider,
    )

    first_provider = next(iter(ensemble_providers.values()))
    placeholder = AIClient(vision_provider=first_provider)
    ensemble_clients: dict[str, Any] = {}
    ensemble_limits: dict[str, Any] = {}
    for family, prov_name in ensemble_providers.items():
        client = AIClient(vision_provider=prov_name)
        conc, qpm, qwin, timeout = vision_limits_for_provider(prov_name)

        class _LimWrap:
            pass

        lim = _LimWrap()
        lim.provider = prov_name
        lim.concurrency = conc
        lim.qpm = qpm
        lim.qpm_window_sec = qwin
        lim.request_timeout_sec = timeout
        ensemble_clients[family] = client
        ensemble_limits[family] = lim

    aesthetic = AestheticClient(
        provider=placeholder,
        defect_provider=placeholder,
        ensemble_providers=ensemble_clients,
        ensemble_limits=ensemble_limits,
    )
    apply_single_config(aesthetic.evaluate._evaluator.config)
    if _STORAGE_ROOT is not None:
        aesthetic.evaluate._evaluator.config.archive_root = str(_STORAGE_ROOT)
    _AESTHETIC_CLIENT = aesthetic
    return aesthetic


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


def score_image_sync(
    *,
    image_bytes: bytes,
    design_requirement: str,
    category: str = "3d",
    image_name: str = "rollout",
) -> tuple[dict[str, Any] | None, str | None]:
    if _AESTHETIC_CLIENT is None:
        raise RuntimeError("aesthetic-core 未配置，请先 build_aesthetic_client()")

    async def _score():
        return await _AESTHETIC_CLIENT.evaluate.comprehensive(
            images=image_bytes,
            design_requirement=design_requirement,
            category=category,
            image_names=[image_name],
        )

    results, error = _run(_score())
    if error:
        return None, error
    if not results:
        return None, "aesthetic comprehensive 无结果"
    row = results[0]
    data = row.model_dump() if hasattr(row, "model_dump") else dict(row)
    return data, None
