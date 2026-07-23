#!/usr/bin/env python3
# 【功能描述】过夜看门狗：等当前批跑结束；未完成则自动并行续跑，直到目标 case 全 ok
# 【输入】与 run_overnight.py 相同的种子列表；轮询间隔
# 【输出】持续更新 outputs/LATEST_DASHBOARD.html；日志 outputs/_overnight_guard.log

from __future__ import annotations

import datetime
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
POLL_SEC = 120               # 检查 overnight 进程 / 完成态间隔
IDLE_BEFORE_RESUME_SEC = 30  # 无 overnight 进程后稍等再启动，避免误抢
MAX_ROUNDS_GUARD = 20        # 最多续跑轮数（防死循环）
# =============================================================================

LOG = _PROJECT_ROOT / "outputs" / "_overnight_guard.log"


def _log(msg: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _overnight_running() -> bool:
    """检测是否有 run_overnight.py 主进程（不含 guard 自身）。"""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Select-Object -ExpandProperty CommandLine"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        errors="ignore",
    )
    text = out.stdout or ""
    for line in text.splitlines():
        if "run_overnight.py" in line and "run_overnight_guard" not in line:
            return True
    return False


def _load_overnight_mod():
    import importlib.util

    path = _SCRIPT_DIR / "run_overnight.py"
    spec = importlib.util.spec_from_file_location("run_overnight_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _target_case_indices() -> list[int]:
    from promptopt.cases import pick_random_synth_prompt

    mod = _load_overnight_mod()
    lib = _PROJECT_ROOT / mod.PROMPT_LIBRARY
    idxs: list[int] = []
    if not mod.SKIP_FIRST_CASE:
        idxs.append(int(mod.FIRST_CASE_INDEX))
    for seed in mod.BATCH_CASE_SEEDS:
        idxs.append(int(pick_random_synth_prompt(seed=seed, path=lib)["index"]))
    return idxs


def _remaining() -> list[int]:
    mod = _load_overnight_mod()
    done = mod.collect_done_case_indices()
    return [i for i in _target_case_indices() if i not in done]


def main() -> None:
    _log(f"guard start · poll={POLL_SEC}s")
    for wave in range(1, MAX_ROUNDS_GUARD + 1):
        # 等当前 overnight 跑完
        while _overnight_running():
            rem = _remaining()
            _log(f"wave{wave}: overnight running · remaining≈{len(rem)} {rem}")
            time.sleep(POLL_SEC)

        rem = _remaining()
        if not rem:
            _log(f"ALL DONE · targets ok · wave={wave}")
            return

        _log(f"wave{wave}: overnight idle · remaining={rem} · resume in {IDLE_BEFORE_RESUME_SEC}s")
        time.sleep(IDLE_BEFORE_RESUME_SEC)
        if _overnight_running():
            _log("another overnight appeared · wait")
            continue

        _log(f"wave{wave}: launching run_overnight.py for remaining {rem}")
        proc = subprocess.Popen(
            [sys.executable, str(_SCRIPT_DIR / "run_overnight.py")],
            cwd=str(_PROJECT_ROOT),
            stdout=open(_PROJECT_ROOT / "outputs" / "_overnight_stdout.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        _log(f"wave{wave}: overnight pid={proc.pid}")
        proc.wait()
        _log(f"wave{wave}: overnight exit={proc.returncode}")

    rem = _remaining()
    _log(f"guard stop · max waves · still remaining={rem}")
    raise SystemExit(1 if rem else 0)


if __name__ == "__main__":
    main()
