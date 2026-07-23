#!/usr/bin/env python3
# 【功能描述】从 overnight manifest 重新生成 Tab 仪表盘（LATEST_DASHBOARD.html）
# 【输入】outputs/overnight_*/manifest.json
# 【输出】outputs/LATEST_DASHBOARD.html + session/dashboard.html

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUTS = _PROJECT_ROOT / "outputs"
REPORT_HTTP_PORT = 9788


def main() -> None:
    from promptopt.report_trace import write_session_dashboard

    sessions = sorted(OUTPUTS.glob("overnight_*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        print("no manifest found")
        sys.exit(1)
    manifest_path = sessions[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dash = write_session_dashboard(manifest_path.parent, manifest)
    print(f"written {dash}")
    print(f"mirror {OUTPUTS / 'LATEST_DASHBOARD.html'}")
    print(f"http://127.0.0.1:{REPORT_HTTP_PORT}/LATEST_DASHBOARD.html")
    print(f"({manifest.get('completed')}/{manifest.get('total')})")


if __name__ == "__main__":
    main()
