"""【功能描述】ReflACT 提示词加载：从 `.md` 文件按名称加载，支持环境级覆盖与通用回退。
【输入】提示词名称 `name`、可选环境名 `env`。
【输出】提示词 Markdown 字符串；`clear_cache()` 清空文件缓存。

- **通用** 提示词：`skillopt/prompts/*.md`
- **环境专属** 提示词：`skillopt/envs/<env>/prompts/*.md`

`load_prompt(name, env)` 先查环境路径，再回退到通用默认。
"""
from __future__ import annotations

import os

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFLACT_DIR = os.path.dirname(_PROMPTS_DIR)

_cache: dict[str, str] = {}


def _read_file(path: str) -> str | None:
    if path in _cache:
        return _cache[path]
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _cache[path] = content
    return content


def load_prompt(name: str, env: str | None = None) -> str:
    """按名称加载提示词，支持环境覆盖与通用回退。

    查找顺序：
      1. ``skillopt/envs/{env}/prompts/{name}.md``  （若提供 *env*）
      2. ``skillopt/prompts/{name}.md``              （通用默认）

    若两处均不存在则抛出 ``FileNotFoundError``。
    """
    if env is not None:
        env_path = os.path.join(_REFLACT_DIR, "envs", env, "prompts", f"{name}.md")
        content = _read_file(env_path)
        if content is not None:
            return content

    generic_path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    content = _read_file(generic_path)
    if content is not None:
        return content

    searched = []
    if env is not None:
        searched.append(os.path.join("skillopt/envs", env, "prompts", f"{name}.md"))
    searched.append(f"skillopt/prompts/{name}.md")
    raise FileNotFoundError(
        f"Prompt '{name}' not found. Searched: {', '.join(searched)}"
    )


def clear_cache() -> None:
    """清空提示词文件缓存（测试时有用）。"""
    _cache.clear()
