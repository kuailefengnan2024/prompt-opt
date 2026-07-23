#!/usr/bin/env python3
# 【功能描述】为已有 run 目录重建全流程 Trace 报告（与训练结束自动生成的 report.html 同格式）
# 【输入】顶部 RUN_DIR
# 【输出】outputs/<run_id>/report.html + pipeline_trace.html

from __future__ import annotations

import sys
from pathlib import Path

# =============================================================================
RUN_DIR = Path(r"D:\prompt-opt\outputs\t2i_20260722_135533")
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from promptopt.report_trace import generate_run_report

    path = generate_run_report(RUN_DIR)
    print(f"http://127.0.0.1:9788/{RUN_DIR.name}/report.html")
    print(f"written {path}")


if __name__ == "__main__":
    main()
