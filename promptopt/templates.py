# 【功能描述】T2I 提示词模板加载与 {占位符} 填充
# 【输入】模板名、字段 dict
# 【输出】填充后的 LLM 用户消息全文

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_cache: dict[str, str] = {}


def has_prompt(name: str) -> bool:
    return (_PROMPTS_DIR / f"{name}.md").is_file()


def load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    key = str(path)
    if key in _cache:
        return _cache[key]
    if not path.is_file():
        raise FileNotFoundError(f"提示词模板不存在: {path}")
    text = path.read_text(encoding="utf-8")
    _cache[key] = text
    return text


def fill_prompt(name: str, fields: dict[str, str]) -> str:
    content = load_prompt(name)
    for key, value in fields.items():
        content = content.replace(f"{{{key}}}", (value or "").strip())
    return content


def get_prompt_scope_section() -> str:
    """语义分层 SSOT，供 Reflect / Merge 注入（与输入是否三段式无关）。"""
    if has_prompt("prompt_scope"):
        return load_prompt("prompt_scope").strip()
    return ""


def list_prompts() -> list[str]:
    if not _PROMPTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))


def clear_cache() -> None:
    _cache.clear()
