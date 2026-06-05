"""【功能描述】Benchmark 数据加载器模板：复制本文件并实现 TODO 以加载 benchmark 数据。
【输入】数据目录 `data_dir`、配置 `cfg`（split_mode、比例等）。
【输出】按 train/valid/test 划分的样本列表，供训练循环通过 `get_split_items` 取用。

`DataLoader` 负责：
1. 从磁盘加载原始数据
2. 划分为 train / validation / test
3. 向训练循环提供 DataItem 对象
"""
from pathlib import Path


class TemplateBenchmarkLoader:
    """<你的 Benchmark 名称> 的数据加载器。

    重命名本类并实现下方方法。
    """

    def __init__(self, data_dir: str = "data/your_benchmark", **kwargs):
        self.data_dir = Path(data_dir)
        self.items = []
        self.splits = {}

    def setup(self, cfg: dict):
        """使用配置初始化加载器。

        训练开始前调用一次。

        Args:
            cfg: 含 `split_mode`、`train_ratio`、`val_ratio` 等键的字典。
        """
        # 步骤 1：加载原始数据
        self.items = self._load_items()

        # 步骤 2：创建划分
        split_mode = cfg.get("split_mode", "ratio")
        if split_mode == "ratio":
            self._split_by_ratio(
                train_ratio=cfg.get("train_ratio", 0.7),
                val_ratio=cfg.get("val_ratio", 0.15),
            )
        elif split_mode == "split_dir":
            self._load_predefined_splits(cfg.get("split_dir", self.data_dir))

    def _load_items(self) -> list:
        """将原始数据加载为结构化条目。

        TODO: 实现数据加载。每条至少包含：
        - id: 唯一标识
        - input: 任务输入（问题、指令等）
        - ground_truth: 期望答案
        - metadata: 可选附加信息字典

        示例:
            items = []
            for path in self.data_dir.glob("*.json"):
                data = json.loads(path.read_text())
                for entry in data:
                    items.append({
                        "id": entry["id"],
                        "input": entry["question"],
                        "ground_truth": entry["answer"],
                        "metadata": {"source": path.name},
                    })
            return items
        """
        raise NotImplementedError("Implement _load_items() for your benchmark")

    def _split_by_ratio(self, train_ratio: float, val_ratio: float):
        """按比例划分条目。"""
        import random
        random.shuffle(self.items)
        n = len(self.items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        self.splits = {
            "train": self.items[:n_train],
            "valid": self.items[n_train:n_train + n_val],
            "test": self.items[n_train + n_val:],
        }

    def _load_predefined_splits(self, split_dir):
        """从预划分目录加载。"""
        # TODO: 若 benchmark 已有预定义划分则在此实现
        raise NotImplementedError

    def get_split_items(self, split: str) -> list:
        """返回指定划分的条目列表。

        Args:
            split: ``"train"``、``"valid"`` 或 ``"test"`` 之一

        Returns:
            该划分的 data item 列表
        """
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.splits.keys())}")
        return self.splits[split]
