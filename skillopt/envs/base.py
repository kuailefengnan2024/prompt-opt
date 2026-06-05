"""【功能描述】ReflACT 环境适配器抽象接口：接入新 benchmark/模拟器时实现 `EnvAdapter` 子类。
【输入】扁平化 `cfg`、批次 `BatchSpec`、当前 skill 文档、rollout 输出等。
【输出】环境管理器、rollout 结果 dict 列表、Reflect 阶段 `RawPatch` 列表及任务类型名。

接入新环境示例::

    class MyBenchAdapter(EnvAdapter):
        def build_train_env(self, batch_size, seed, **kw):
            return MyEnvManager(split="train", n=batch_size, seed=seed)

        def build_eval_env(self, env_num, split, seed, **kw):
            return MyEnvManager(split=split, n=env_num, seed=seed)

        def rollout(self, env_manager, skill_content, out_dir, **kw):
            # 运行 episode，返回 [{"id": ..., "hard": 0/1, "soft": 0.0-1.0, ...}]
            ...

        def reflect(self, results, skill_content, out_dir, **kw):
            # 分析轨迹，返回 patch dict 列表
            ...

        def get_task_types(self):
            return ["task_a", "task_b"]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
import random

from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.prompts import load_prompt


class EnvAdapter(ABC):
    """将 ReflACT 接入任意环境的抽象适配器。

    子类必须实现全部抽象方法；ReflACT trainer 在流水线各阶段调用它们。
    """

    # ── 生命周期钩子 ────────────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        """trainer 在训练循环开始前调用一次。

        可覆盖以做需要完整配置的一次性初始化（如数据加载、划分创建）。默认为空操作。
        """
        self._cfg = dict(cfg)

    def get_dataloader(self) -> BaseDataLoader | None:
        """返回本适配器使用的任务 dataloader（若有）。"""
        return None

    def requires_ray(self) -> bool:
        """返回本适配器是否需要初始化 Ray 运行时。"""
        return False

    def build_reference_text(self, item: dict) -> str:
        """返回用于反思的隐藏参考材料（若有）。"""
        return str(item.get("reference_text") or "").strip()

    def get_reference_metadata(self, item: dict) -> dict:
        """返回隐藏参考材料的结构化元数据。"""
        reference_text = self.build_reference_text(item)
        if not reference_text:
            return {"fields": [], "preview": ""}
        return {
            "fields": ["reference_text"],
            "preview": reference_text[:400],
        }

    def attach_reference_context(
        self,
        results: list[dict],
        items: list[dict] | None,
    ) -> list[dict]:
        """将环境专有的隐藏参考文本附加到结果 dict 上。"""
        if not results or not items:
            return list(results)

        item_by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, dict) and item.get("id") is not None
        }
        enriched: list[dict] = []
        for row in results:
            merged = dict(row)
            item = item_by_id.get(str(row.get("id")))
            if item:
                reference_text = self.build_reference_text(item)
                if reference_text:
                    merged["reference_text"] = reference_text
            enriched.append(merged)
        return enriched

    def select_representative_items(
        self,
        results: list[dict],
        items: list[dict] | None,
        *,
        n_failures: int,
        n_successes: int,
        seed: int | None = None,
    ) -> list[dict]:
        """按成败从当前批次中选取少量多样化代表条目。"""
        if not items:
            return []

        item_by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, dict) and item.get("id") is not None
        }
        failures = [
            (result, item_by_id[str(result.get("id"))])
            for result in results
            if not result.get("hard") and str(result.get("id")) in item_by_id
        ]
        successes = [
            (result, item_by_id[str(result.get("id"))])
            for result in results
            if result.get("hard") and str(result.get("id")) in item_by_id
        ]

        rng = random.Random(seed)

        def _pick(pool: list[tuple[dict, dict]], quota: int) -> list[dict]:
            if quota <= 0 or not pool:
                return []
            shuffled = list(pool)
            rng.shuffle(shuffled)

            picked_ids: set[str] = set()
            picked: list[dict] = []
            seen_types: set[str] = set()

            for result, item in shuffled:
                task_type = str(result.get("task_type") or item.get("task_type") or item.get("subtype") or "unknown")
                item_id = str(item["id"])
                if task_type in seen_types or item_id in picked_ids:
                    continue
                picked.append(item)
                picked_ids.add(item_id)
                seen_types.add(task_type)
                if len(picked) >= quota:
                    return picked

            for _, item in shuffled:
                item_id = str(item["id"])
                if item_id in picked_ids:
                    continue
                picked.append(item)
                picked_ids.add(item_id)
                if len(picked) >= quota:
                    break
            return picked

        selected = _pick(failures, n_failures)
        selected_ids = {str(item["id"]) for item in selected}
        selected.extend(
            item for item in _pick(successes, n_successes)
            if str(item["id"]) not in selected_ids
        )
        return selected

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        """从 :class:`BatchSpec` 构建环境管理器或条目列表。

        默认行为保持 legacy 适配器 API：train 批次走 :meth:`build_train_env`，
        eval 批次走 :meth:`build_eval_env`。
        """
        if batch.phase == "train":
            return self.build_train_env(batch_size=batch.batch_size, seed=batch.seed, **kwargs)
        return self.build_eval_env(
            env_num=batch.batch_size,
            split=batch.split,
            seed=batch.seed,
            **kwargs,
        )

    @abstractmethod
    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        """构建训练用环境管理器。

        Returns
        -------
        object
            可传入 :meth:`rollout` 的环境管理器。
        """

    @abstractmethod
    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        """构建评估用环境管理器。

        Parameters
        ----------
        env_num : int
            评估环境数量。
        split : str
            数据集划分（如 ``"valid_seen"``、``"valid_unseen"``）。
        seed : int
            可复现的随机种子。

        Returns
        -------
        object
            可传入 :meth:`rollout` 的环境管理器。
        """

    @abstractmethod
    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        """使用当前 skill 运行一批 episode。

        Returns
        -------
        list[dict]
            每个 dict 符合 :class:`~skillopt.types.RolloutResult`：
            须含 ``"id"`` (str)、``"hard"`` (0/1)、``"soft"`` (float 0-1)；
            可含环境专有字段。
        """

    @abstractmethod
    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        """分析 rollout 结果并生成 patch。

        每个返回 dict 符合 :class:`~skillopt.types.RawPatch`：
        ``"patch"``（含 ``"edits"`` 列表）+ ``"source_type"``
        （``"failure"`` 或 ``"success"``）。

        Returns
        -------
        list[dict | None]
            原始分析师输出；``None`` 条目会被过滤。
        """

    @abstractmethod
    def get_task_types(self) -> list[str]:
        """返回本环境的任务类型名称列表。"""

    # ── 提示词配置（两级优先级）────────────────────────────────────────────
    #
    # 优先级：环境专属提示词文件 > 通用默认提示词文件。
    #
    # 通过 ``load_prompt(name, env)`` 从 ``.md`` 加载：
    #   1. ``skillopt/envs/<env>/prompts/<name>.md``  （环境专属）
    #   2. ``skillopt/prompts/<name>.md``             （通用回退）
    #
    # 子类仍可覆盖 ``get_*_prompt()`` 以完全自定义。

    @property
    def _env_name(self) -> str:
        """从本适配器模块路径推导 env 目录名。"""
        # 例: "skillopt.envs.searchqa.adapter" → "searchqa"
        module = type(self).__module__
        parts = module.split(".")
        if len(parts) >= 3 and parts[-3] == "envs":
            return parts[-2]
        return ""

    def _load_env_prompt(self, name: str) -> str | None:
        """加载带环境覆盖的提示词；未找到时返回 None。"""
        try:
            return load_prompt(name, env=self._env_name)
        except FileNotFoundError:
            return None

    def get_error_minibatch_prompt(self) -> str | None:
        update_mode = getattr(self, "_cfg", {}).get("skill_update_mode", "patch")
        raw_mode = str(update_mode).strip().lower()
        if raw_mode in {"full_rewrite", "full_rewrite_minibatch", "minibatch_full_rewrite", "skill_rewrite_minibatch"}:
            prompt = self._load_env_prompt("analyst_error_full_rewrite")
            if prompt is not None:
                return prompt
        if raw_mode in {"rewrite", "rewrite_from_suggestions", "suggestions", "rewrite_suggestions"}:
            prompt = self._load_env_prompt("analyst_error_rewrite")
            if prompt is not None:
                return prompt
        return self._load_env_prompt("analyst_error")

    def get_success_minibatch_prompt(self) -> str | None:
        update_mode = getattr(self, "_cfg", {}).get("skill_update_mode", "patch")
        raw_mode = str(update_mode).strip().lower()
        if raw_mode in {"full_rewrite", "full_rewrite_minibatch", "minibatch_full_rewrite", "skill_rewrite_minibatch"}:
            prompt = self._load_env_prompt("analyst_success_full_rewrite")
            if prompt is not None:
                return prompt
        if raw_mode in {"rewrite", "rewrite_from_suggestions", "suggestions", "rewrite_suggestions"}:
            prompt = self._load_env_prompt("analyst_success_rewrite")
            if prompt is not None:
                return prompt
        return self._load_env_prompt("analyst_success")
