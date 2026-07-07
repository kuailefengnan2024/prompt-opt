# 【功能描述】外部引擎桥接层（api-core / aesthetic-core），引擎包内不 import 实现细节
# 【输入】runtime_config 与宿主注入的 provider 名
# 【输出】llm_chat_sync、generate_image_sync、score_image_sync 等同步 API

from .api_core_bridge import generate_image_sync, llm_chat_sync
from .aesthetic_bridge import build_aesthetic_client, score_image_sync

__all__ = [
    "llm_chat_sync",
    "generate_image_sync",
    "build_aesthetic_client",
    "score_image_sync",
]
