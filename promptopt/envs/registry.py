"""【功能描述】环境适配器注册表：按 `env.name` 解析并实例化 `EnvAdapter`。
【输入】扁平化配置 dict（环境名及适配器构造参数）。
【输出】`EnvAdapter` 实例；新增 benchmark 时向 `_BUILTIN_ENVS` 追加条目。
"""
from __future__ import annotations

import inspect

_ENV_REGISTRY: dict[str, type] = {}
_LOADED = False

# (env_name, import_path, class_name)
_BUILTIN_ENVS: list[tuple[str, str, str]] = [
    ("t2i", "promptopt.envs.t2i.adapter", "T2IAdapter"),
]


def _register_builtins() -> None:
    global _LOADED
    if _LOADED:
        return
    for env_name, module_path, class_name in _BUILTIN_ENVS:
        try:
            module = __import__(module_path, fromlist=[class_name])
            _ENV_REGISTRY[env_name] = getattr(module, class_name)
        except ImportError:
            pass
    _LOADED = True


def register_env(name: str, adapter_cls: type) -> None:
    """运行时注册自定义环境（用于 benchmark 扩展）。"""
    _ENV_REGISTRY[name] = adapter_cls


def list_envs() -> list[str]:
    _register_builtins()
    return sorted(_ENV_REGISTRY.keys())


def get_adapter(cfg: dict):
    _register_builtins()
    env_name = str(cfg.get("env") or "t2i")
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list_envs()}. "
            f"Add promptopt/envs/{env_name}/ or register via register_env()."
        )
    adapter_cls = _ENV_REGISTRY[env_name]
    sig = inspect.signature(adapter_cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    adapter_kwargs = {k: cfg[k] for k in accepted if k in cfg}
    return adapter_cls(**adapter_kwargs)
