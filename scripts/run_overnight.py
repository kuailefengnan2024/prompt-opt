#!/usr/bin/env python3
# 【功能描述】过夜串联：1 个跑满 8 轮 + 10 个 batch case，生成报告索引页
# 【输入】本文件顶部配置
# 【输出】outputs/overnight_<ts>/manifest.json + Tab 仪表盘；各 t2i_<ts>/report.html（全流程 Trace）

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# ★ 过夜任务配置
# =============================================================================

FIRST_CASE_INDEX = 70        # 接续上次 case #70（指尖非遗馆），跑满 8 轮
BATCH_CASE_SEEDS = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]  # 后续 10 个
RUN_FIRST_ONLY = True        # True：只跑 case70 验证；通过后改 False 跑满后续
PROMPT_LIBRARY = "data/kv_synth_prompts_100_only.json"

MAX_ROUNDS = 8
TRAIN_RUNS = 4
GATE_RUNS = 2
EDIT_BUDGET = 6
CATEGORY = "3d"
HARD_THRESHOLD = 65.0
MIN_ENSEMBLE_CONFIDENCE = 0.5
REASON_MIN_DIM_CONFIDENCE = 0.833
LLM_PROVIDER = "doubao21pro"
IMAGE_PROVIDER = "seedream_4_5"
IMAGE_SIZE = "2999x1687"
AESTHETIC_ENSEMBLE = {"doubao": "doubao_seed_2_0_pro_vision"}
MINIBATCH_SIZE = 8
ANALYST_WORKERS = 4
MERGE_BATCH_SIZE = 4
SEED = 42
USE_META_PROMPT = True
META_PROMPT_MAX_CHARS = 1500
SAVE_DEBUG = False
# 设计要求：直接填入；空则从 prompt 主副标题/意象摘要解析
DESIGN_REQUIREMENT = ""

MAX_ATTEMPTS = 3              # 单 run 失败重试次数
RETRY_SLEEP_SEC = 120         # 重试间隔（应对 Connection error）

# =============================================================================

_AESTHETIC_ROOT = Path(r"D:\kv-generator\engines\aesthetic-core")
if _AESTHETIC_ROOT.exists() and str(_AESTHETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AESTHETIC_ROOT))


def _load_dotenv() -> None:
    for path in (_PROJECT_ROOT / ".env", Path(r"D:\api-core\.env"), Path(r"D:\kv-generator\.env")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.startswith("export "):
                key = key[7:].strip()
            os.environ.setdefault(key, val)


def _build_cfg(*, case: dict, out_root: str):
    from promptopt.engine.runner import T2IRunConfig

    design_req = (DESIGN_REQUIREMENT or "").strip()
    if design_req and not case.get("design_requirement"):
        case = {**case, "design_requirement": design_req}
    return T2IRunConfig(
        initial_prompt=case["prompt"],
        category=CATEGORY,
        max_rounds=MAX_ROUNDS,
        train_runs=TRAIN_RUNS,
        gate_runs=GATE_RUNS,
        edit_budget=EDIT_BUDGET,
        hard_threshold=HARD_THRESHOLD,
        min_ensemble_confidence=MIN_ENSEMBLE_CONFIDENCE,
        reason_min_dim_confidence=REASON_MIN_DIM_CONFIDENCE,
        llm_provider=LLM_PROVIDER,
        image_provider=IMAGE_PROVIDER,
        image_size=IMAGE_SIZE,
        aesthetic_ensemble=AESTHETIC_ENSEMBLE,
        minibatch_size=MINIBATCH_SIZE,
        analyst_workers=ANALYST_WORKERS,
        merge_batch_size=MERGE_BATCH_SIZE,
        seed=SEED,
        out_root=out_root,
        case_meta=case,
        design_requirement_text=design_req or str(case.get("design_requirement") or ""),
        save_debug=SAVE_DEBUG,
        use_meta_prompt=USE_META_PROMPT,
        meta_prompt_max_chars=META_PROMPT_MAX_CHARS,
    )


def _run_with_retry(cfg, label: str):
    from promptopt.engine.runner import run_t2i_optimize

    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"  [attempt {attempt}/{MAX_ATTEMPTS}] {label}")
            return run_t2i_optimize(cfg)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  [FAIL] attempt {attempt} failed: {exc}")
            traceback.print_exc()
            if attempt < MAX_ATTEMPTS:
                print(f"  … 等待 {RETRY_SLEEP_SEC}s 后重试")
                time.sleep(RETRY_SLEEP_SEC)
    raise last_err  # type: ignore[misc]


def _write_dashboard(session_dir: Path, manifest: dict) -> Path:
    from promptopt.report_trace import write_session_dashboard

    path = write_session_dashboard(session_dir, manifest)
    return Path(path)


def _save_manifest(session_dir: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = session_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _write_dashboard(session_dir, manifest)


def _run_one(
    *,
    session_dir: Path,
    manifest: dict,
    index: int,
    label: str,
    case: dict,
) -> None:
    lib_path = _PROJECT_ROOT / PROMPT_LIBRARY
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = str(_PROJECT_ROOT / "outputs" / f"t2i_{run_id}")
    os.makedirs(out_root, exist_ok=True)

    print("\n" + "#" * 72)
    print(f"[{index}/{manifest['total']}] {label} case_index={case['index']}")
    print(f"out: {out_root}")
    print("#" * 72)

    row: dict = {
        "index": index,
        "label": label,
        "case_index": case.get("index"),
        "out_root": out_root,
        "rounds": MAX_ROUNDS,
    }
    t0 = time.time()
    try:
        cfg = _build_cfg(case=case, out_root=out_root)
        summary = _run_with_retry(cfg, label)
        elapsed = round(time.time() - t0, 1)
        best = summary.get("best") or {}
        initial = summary.get("initial") or {}
        report_path = summary.get("report") or str(Path(out_root) / "report.html")
        row.update({
            "status": "ok",
            "report": report_path,
            "initial_score": initial.get("final_score"),
            "best_score": best.get("final_score"),
            "best_step": best.get("step"),
            "elapsed_s": elapsed,
        })
        print(f"  [OK] done best={row.get('best_score')} step={row.get('best_step')} ({elapsed}s)")
    except Exception as exc:  # noqa: BLE001
        row.update({
            "status": "error",
            "error": str(exc),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  [FAIL] final failure: {exc}")

    manifest["runs"].append(row)
    manifest["completed"] = len(manifest["runs"])
    _save_manifest(session_dir, manifest)


def main() -> None:
    _load_dotenv()
    from promptopt.cases import pick_random_synth_prompt, pick_synth_prompt_by_index

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = _PROJECT_ROOT / "outputs" / f"overnight_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    lib_path = _PROJECT_ROOT / PROMPT_LIBRARY
    jobs: list[tuple[str, str, dict]] = [
        (
            "full_8r",
            "跑满8轮·case70",
            pick_synth_prompt_by_index(FIRST_CASE_INDEX, path=lib_path),
        ),
    ]
    if not RUN_FIRST_ONLY:
        for i, seed in enumerate(BATCH_CASE_SEEDS, 1):
            jobs.append((
                f"batch_{i:02d}",
                f"批量{i}/10",
                pick_random_synth_prompt(seed=seed, path=lib_path),
            ))
    else:
        print("[配置] RUN_FIRST_ONLY=True → 仅跑第一个 case，验证通过后改 False")

    manifest: dict = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "total": len(jobs),
        "completed": 0,
        "max_rounds": MAX_ROUNDS,
        "runs": [],
    }
    _save_manifest(session_dir, manifest)

    log_path = session_dir / "run.log"
    print(f"过夜任务启动 total={len(jobs)} session={session_dir}")
    print(f"dashboard 入口: outputs/LATEST_DASHBOARD.html")

    t_all = time.time()
    for idx, (job_id, label, case) in enumerate(jobs, 1):
        _run_one(
            session_dir=session_dir,
            manifest=manifest,
            index=idx,
            label=f"{job_id}:{label}",
            case=case,
        )

    manifest["total_elapsed_s"] = round(time.time() - t_all, 1)
    _save_manifest(session_dir, manifest)
    print("\n" + "=" * 72)
    print(f"全部结束 {manifest['completed']}/{manifest['total']} · {manifest['total_elapsed_s']}s")
    print(f"索引页: {session_dir / 'dashboard.html'}")
    print(f"固定入口: {_PROJECT_ROOT / 'outputs' / 'LATEST_DASHBOARD.html'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
