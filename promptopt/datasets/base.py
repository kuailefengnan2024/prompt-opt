"""【功能描述】Reflect 通用任务数据加载抽象：批次采样与 episode 规划（非张量 DataLoader）。
【输入】训练配置 `cfg`、划分目录或 `data_path`、batch/seed 等采样参数。
【输出】`BatchSpec` 列表及 train/val/test 条目，供环境与 trainer 消费。

Reflect 不直接训练模型参数，而是迭代任务批次、rollout 当前 prompt、反思并更新 prompt 文档；
因此此处的「dataloader」更接近批次采样器 / episode 规划器。

类层次::

    BaseDataLoader          # 抽象 — 模拟器类环境（如 ALFWorld）
    └── SplitDataLoader     # 抽象 — 带 split_dir 的数据集类环境

SplitDataLoader 支持两种数据集入口：

1. ``split_mode="split_dir"``：消费已有划分目录。
2. ``split_mode="ratio"``：从原始 ``data_path`` 按 train:val:test 比例
   确定性生成划分目录。

标准划分布局为::

    split_dir/
    ├── train/      # 训练样本
    ├── val/        # 验证 / 选择集（gate）
    └── test/       # 留出测试集

各子目录内容为 benchmark 专有；子类仅需实现
``load_split_items(split_path)`` 以说明如何读取其中一个目录。
"""
from __future__ import annotations

import glob
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BatchSpec:
    """训练循环消费的具体批次请求。

    Parameters
    ----------
    phase : str
        ``"train"`` 或 ``"eval"``。
    split : str
        数据集划分名，通常为 ``"train"`` 或某 eval 划分。
    seed : int
        用于确定性构造本批次的随机种子。
    batch_size : int
        本批次请求的条目 / episode 数量。
    payload : object | None
        环境专有的批次载荷。数据集类环境常为采样条目列表；
        模拟器类环境可为 ``None``，仅由 seed 定义批次。
    metadata : dict[str, Any]
        可选结构化元数据，用于日志、恢复或课程学习。
    """

    phase: str
    split: str
    seed: int
    batch_size: int
    payload: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataLoader(ABC):
    """Reflect 任务批次规划的抽象基类。

    子类负责定义 train/eval 批次如何采样；本类默认实现提供
    确定性的 epoch seed 规划，使各 loader 共享相同可复现行为。
    """

    def setup(self, cfg: dict) -> None:
        """可选：使用完整 trainer 配置做一次性初始化。"""

    def set_out_root(self, out_root: str) -> None:
        """可选：供需持久化划分文件或状态的 loader 使用。"""

    def state_dict(self) -> dict[str, Any]:
        """返回可序列化的 loader 状态，用于恢复训练。"""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """从 :meth:`state_dict` 输出恢复 loader 状态。"""

    def get_train_size(self) -> int | None:
        """在已知时返回训练池大小。"""
        return None

    @staticmethod
    def make_base_seeds(steps_per_epoch: int, accumulation: int, seed: int) -> list[int]:
        """返回用于定义 train 批次的确定性种子池。"""
        batches_per_epoch = steps_per_epoch * accumulation
        return [seed + i + 1 for i in range(batches_per_epoch)]

    @staticmethod
    def shuffle_epoch_seeds(base_seeds: list[int], epoch: int, seed: int) -> list[int]:
        """返回 *base_seeds* 在每个 epoch 的确定性打乱结果。"""
        epoch_rng = random.Random(seed + epoch * 1000)
        shuffled = list(base_seeds)
        epoch_rng.shuffle(shuffled)
        return shuffled

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        """构建一个 epoch 内的全部 train 批次列表。"""
        base_seeds = self.make_base_seeds(
            steps_per_epoch=steps_per_epoch,
            accumulation=accumulation,
            seed=seed,
        )
        shuffled_seeds = self.shuffle_epoch_seeds(base_seeds, epoch=epoch, seed=seed)
        return [
            self.build_train_batch(batch_size=batch_size, seed=batch_seed, **kwargs)
            for batch_seed in shuffled_seeds
        ]

    @abstractmethod
    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        """构造一条 train 批次规格。"""

    @abstractmethod
    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        """构造一条 eval 批次规格。"""


# ── 数据集类环境的划分式 DataLoader ─────────────────────────────────────

# split_dir/ 下期望的标准划分名
SPLIT_NAMES = ("train", "val", "test")

# legacy / trainer 划分名 → 标准目录名
_SPLIT_ALIAS: dict[str, str] = {
    "train": "train",
    "valid_seen": "val",
    "selection": "val",
    "val": "val",
    "valid_unseen": "test",
    "test": "test",
}


def _load_json_or_jsonl(path: str) -> list[dict]:
    """从 JSON 或 JSONL 文件加载条目列表。"""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        nested = data.get("data")
        if isinstance(nested, list):
            return nested
        return list(data.values())

    items: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _parse_split_ratio(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(text or "").split(":") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"split_ratio must be in train:val:test form, got {text!r}"
        )
    try:
        train, val, test = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"split_ratio must contain integers, got {text!r}"
        ) from exc
    if min(train, val, test) <= 0:
        raise ValueError(f"split_ratio parts must be positive, got {text!r}")
    return train, val, test


def _compute_split_counts(total: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    weights = list(ratio)
    denom = sum(weights)
    raw = [total * weight / denom for weight in weights]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - counts[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


class SplitDataLoader(BaseDataLoader):
    """数据集类环境的基类。

    支持模式：

    - ``split_mode="split_dir"``：加载已有 ``train/``、``val/``、``test/`` 目录树。
    - ``split_mode="ratio"``：从 ``data_path`` 加载原始条目并按比例
      确定性物化划分目录。
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        self.split_dir = split_dir
        self.data_path = data_path
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.split_seed = int(split_seed)
        self.split_output_dir = split_output_dir
        self.seed = seed
        self.limit = limit
        self._splits: dict[str, list[dict]] = {}

    # ── 初始化 ────────────────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        if not self.split_mode:
            self.split_mode = str(cfg.get("split_mode", "ratio") or "ratio")
        if not self.split_dir:
            self.split_dir = cfg.get("split_dir", "")
        if not self.data_path:
            self.data_path = cfg.get("data_path", "")
        if not self.split_output_dir:
            self.split_output_dir = cfg.get("split_output_dir", "")
        if "split_seed" in cfg and not self.split_seed:
            self.split_seed = int(cfg.get("split_seed", 0) or 0)
        if not self.split_seed:
            self.split_seed = self.seed
        if not self.split_ratio:
            self.split_ratio = str(cfg.get("split_ratio", "2:1:7") or "2:1:7")

        mode = str(self.split_mode or "ratio").strip().lower()
        if mode not in {"ratio", "split_dir"}:
            raise ValueError(
                f"{type(self).__name__} split_mode must be 'ratio' or 'split_dir', "
                f"got {self.split_mode!r}"
            )
        self.split_mode = mode

        if self.split_mode == "ratio":
            self.split_dir = self._materialize_ratio_split(cfg)
        if not self.split_dir:
            raise ValueError(
                f"{type(self).__name__} requires either "
                "`split_mode=ratio` with `data_path`, or `split_mode=split_dir` "
                f"with `split_dir` pointing to {'/'.join(SPLIT_NAMES)}/."
            )
        self._load_all_splits()

    def _resolve_split_output_dir(self, cfg: dict) -> str:
        if self.split_output_dir:
            return os.path.abspath(self.split_output_dir)
        out_root = os.path.abspath(str(cfg.get("out_root") or os.getcwd()))
        env_name = str(cfg.get("env") or type(self).__name__.replace("DataLoader", "").lower())
        ratio_tag = str(self.split_ratio or "2:1:7").replace(":", "-")
        return os.path.join(out_root, "_generated_splits", f"{env_name}_{ratio_tag}_seed{self.split_seed}")

    def load_raw_items(self, data_path: str) -> list[dict]:
        """在按比例划分前从数据集路径加载原始条目。

        当原始数据不是单个 JSON/JSONL 或目录布局需自定义规范化时，子类可覆盖。
        """
        if os.path.isdir(data_path):
            if any(os.path.isdir(os.path.join(data_path, name)) for name in SPLIT_NAMES):
                raise ValueError(
                    f"{type(self).__name__} got a split directory as data_path. "
                    "Use split_mode=split_dir and pass it as split_dir instead."
                )
            candidates = sorted(glob.glob(os.path.join(data_path, "*.json")))
            candidates += sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
            if len(candidates) != 1:
                raise ValueError(
                    f"{type(self).__name__} expected data_path to be one JSON/JSONL file "
                    f"or a directory containing exactly one such file, got: {data_path}"
                )
            return _load_json_or_jsonl(candidates[0])
        return _load_json_or_jsonl(data_path)

    def write_split_items(self, split_path: str, items: list[dict]) -> None:
        os.makedirs(split_path, exist_ok=True)
        out_path = os.path.join(split_path, "items.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _materialize_ratio_split(self, cfg: dict) -> str:
        data_path = os.path.abspath(str(self.data_path or "").strip())
        if not data_path:
            raise ValueError(
                f"{type(self).__name__} requires data_path when split_mode=ratio."
            )

        ratio = _parse_split_ratio(self.split_ratio)
        items = self.load_raw_items(data_path)
        if not isinstance(items, list) or not items:
            raise ValueError(f"No raw items available for ratio split from {data_path}")

        shuffled = list(items)
        rng = random.Random(self.split_seed)
        rng.shuffle(shuffled)

        train_n, val_n, test_n = _compute_split_counts(len(shuffled), ratio)
        train_items = shuffled[:train_n]
        val_items = shuffled[train_n: train_n + val_n]
        test_items = shuffled[train_n + val_n: train_n + val_n + test_n]

        split_dir = self._resolve_split_output_dir(cfg)
        manifest = {
            "source_data_path": data_path,
            "split_mode": "ratio",
            "split_ratio": self.split_ratio,
            "split_seed": self.split_seed,
            "counts": {
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            },
        }
        os.makedirs(split_dir, exist_ok=True)
        self.write_split_items(os.path.join(split_dir, "train"), train_items)
        self.write_split_items(os.path.join(split_dir, "val"), val_items)
        self.write_split_items(os.path.join(split_dir, "test"), test_items)
        with open(os.path.join(split_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(
            f"  [{type(self).__name__}] generated ratio split {self.split_ratio} "
            f"at {split_dir} from {data_path}"
        )
        return split_dir

    def _load_all_splits(self) -> None:
        for name in SPLIT_NAMES:
            split_path = os.path.join(self.split_dir, name)
            if not os.path.isdir(split_path):
                raise ValueError(
                    f"Missing '{name}/' subdirectory in split_dir: {self.split_dir}"
                )
            items = self.load_split_items(split_path)
            if self.limit:
                items = items[: self.limit]
            self._splits[name] = items

        counts = " ".join(f"{k}={len(v)}" for k, v in self._splits.items())
        print(f"  [{type(self).__name__}] {counts}  (from {self.split_dir})")

    def load_split_items(self, split_path: str) -> list[dict]:
        """从某一划分目录加载条目（如 ``split_dir/train/``）。

        默认：在目录中找第一个 ``.json`` 并按 JSON 数组加载。
        子类可覆盖以支持自定义格式。
        """
        json_files = sorted(glob.glob(os.path.join(split_path, "*.json")))
        if not json_files:
            raise FileNotFoundError(
                f"No .json file found in {split_path}"
            )
        with open(json_files[0], encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(
                f"Expected JSON array in {json_files[0]}, got {type(items).__name__}"
            )
        return items

    # ── 访问器 ────────────────────────────────────────────────────────────

    @property
    def train_items(self) -> list[dict]:
        return self._splits.get("train", [])

    @property
    def val_items(self) -> list[dict]:
        return self._splits.get("val", [])

    @property
    def test_items(self) -> list[dict]:
        return self._splits.get("test", [])

    def get_split_items(self, split: str) -> list[dict]:
        """将划分名（含 legacy 别名）解析为对应条目列表。"""
        canonical = _SPLIT_ALIAS.get(split, split)
        return list(self._splits.get(canonical, self.val_items))

    def get_train_size(self) -> int:
        return len(self.train_items)

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        """构建覆盖 train 划分一次打乱遍历的完整 epoch。

        对划分型数据集，一个 epoch 应对可用训练条目做一次遍历，
        而非重复独立随机采样。
        """
        epoch_rng = random.Random(seed + epoch * 1000)
        items = list(self.train_items)
        epoch_rng.shuffle(items)

        total_batches = steps_per_epoch * accumulation
        if total_batches <= 0:
            return []

        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            batch_items = items[cursor: cursor + batch_size]
            cursor += len(batch_items)

            # 极小数据集在 accumulation > 1 时可能产生尾部空 microbatch；
            # 此时复用打乱前缀，使 trainer 仍收到预期批次数。
            if not batch_items and items:
                refill_rng = random.Random(seed + epoch * 1000 + batch_idx + 1)
                batch_items = list(items)
                refill_rng.shuffle(batch_items)
                batch_items = batch_items[:batch_size]

            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=seed + epoch * 1000 + batch_idx + 1,
                    batch_size=len(batch_items),
                    payload=batch_items,
                )
            )

        return batches

    # ── 批次构造 ─────────────────────────────────────────────────────────

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        rng = random.Random(seed)
        items = list(self.train_items)
        rng.shuffle(items)
        items = items[:batch_size]
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        items = self.get_split_items(split)
        if env_num and env_num < len(items):
            items = items[:env_num]
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )
