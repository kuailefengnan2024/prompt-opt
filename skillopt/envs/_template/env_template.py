"""【功能描述】Benchmark 环境适配器模板：复制本文件并实现 TODO 以接入新 benchmark。
【输入】配置 `cfg`、单条任务 `item`、当前 skill 文档、target `model` 等。
【输出】结构化任务结果（prediction、score、trajectory 等），供训练循环消费。

`EnvAdapter` 负责：
1. 使用 target 模型 + 当前 skill 文档执行任务
2. 将预测与 ground truth 对比评估
3. 向训练循环返回结构化结果
"""
from skillopt.envs.base import EnvAdapter


class TemplateBenchmarkEnv(EnvAdapter):
    """<你的 Benchmark 名称> 的环境适配器。

    重命名本类并实现下方抽象方法。
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # TODO: 初始化 benchmark 专有状态
        # 示例: self.tools = load_tools(cfg)

    async def execute(self, item, skill: str, model):
        """使用 target 模型执行单条任务。

        Args:
            item: DataItem，含 .id、.input、.ground_truth、.metadata
            skill: 当前 skill 文档内容（Markdown 字符串）
            model: target 模型后端实例

        Returns:
            TaskResult，含 prediction、score、trajectory
        """
        # 步骤 1：组合 skill 与任务输入构建 prompt
        prompt = self.build_prompt(item, skill)

        # 步骤 2：调用 target 模型
        # TODO: 按 benchmark 定制消息格式
        messages = [
            {"role": "system", "content": skill},
            {"role": "user", "content": item.input},
        ]
        response = await model.generate(messages)

        # 步骤 3：将模型响应解析为 prediction
        prediction = self.parse_response(response.content)

        # 步骤 4：对 prediction 打分
        score = self.evaluate(prediction, item.ground_truth)

        # 步骤 5：返回结构化结果
        return {
            "item_id": item.id,
            "prediction": prediction,
            "score": score,
            "trajectory": messages + [{"role": "assistant", "content": response.content}],
        }

    def evaluate(self, prediction: str, ground_truth: str) -> float:
        """将 prediction 与 ground truth 对比打分。

        Returns:
            0.0（错）到 1.0（对）之间的浮点数

        TODO: 实现你的评分指标。常见选项：
        - 精确匹配: float(pred.strip().lower() == gt.strip().lower())
        - F1: token 重叠
        - ANLS: 文档 QA
        - 自定义: [0, 1] 内任意浮点
        """
        # 占位 — 精确匹配
        return float(prediction.strip().lower() == ground_truth.strip().lower())

    def build_prompt(self, item, skill: str) -> str:
        """将 skill 文档与任务输入组合。"""
        return f"{skill}\n\n---\n\nQuestion: {item.input}"

    def parse_response(self, response: str) -> str:
        """从模型原始响应中提取答案。

        TODO: 实现提取逻辑，例如：
        - 提取 "Answer:" 之后的文本
        - 解析 JSON 输出
        - 从代码块中提取
        """
        return response.strip()
