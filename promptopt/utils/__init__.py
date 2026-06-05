"""【功能描述】Reflect 工具模块：JSON 解析、打分与哈希等通用辅助。
【输入】LLM 响应文本、episode 结果列表、prompt 文档字符串等。
【输出】`extract_json` / `extract_json_array`、`compute_score`、`prompt_hash` 等函数再导出。
"""

from promptopt.utils.json_utils import extract_json, extract_json_array  # noqa: F401
from promptopt.utils.scoring import compute_score, prompt_hash  # noqa: F401
