#!/usr/bin/env python3
# 【功能描述】过夜批跑：多 case 多进程并行 Reflect 优化，写 Tab 仪表盘
# 【输入】本文件顶部配置（PARALLEL_JOBS / BATCH_CASE_SEEDS 等）
# 【输出】outputs/overnight_<ts>/manifest.json + dashboard；各 t2i_*/report.html

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# ★ 过夜任务配置
# =============================================================================

FIRST_CASE_INDEX = 70
BATCH_CASE_SEEDS = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]
RUN_FIRST_ONLY = False
SKIP_FIRST_CASE = True       # case70 已完成则跳过
AUTO_SKIP_DONE = True        # 已有 ok 报告的 case_index 自动跳过（便于续跑）
# 只认此时间之后的成功产物（避免旧 overnight 误判「已完成」）
DONE_SINCE = "2026-07-23T16:00:00"
PROMPT_LIBRARY = "data/kv_synth_prompts_100_only.json"

PARALLEL_JOBS = 3            # 同时跑几个 case（多进程）；过大易打爆 API / 连接错误

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
ANALYST_WORKERS = 2          # 并行时略降，避免单机 Reflect 线程爆炸
MERGE_BATCH_SIZE = 4
SEED = 42
USE_META_PROMPT = True
META_PROMPT_MAX_CHARS = 1500
SAVE_DEBUG = False
DESIGN_REQUIREMENT = ""

MAX_ATTEMPTS = 3
RETRY_SLEEP_SEC = 120

# =============================================================================

_AESTHETIC_ROOT = Path(r"D:\kv-generator\engines\aesthetic-core")


def _ensure_paths() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
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
            print(f"  [{label}] attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
            return run_t2i_optimize(cfg)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  [{label}] FAIL attempt {attempt}: {exc}", flush=True)
            traceback.print_exc()
            if attempt < MAX_ATTEMPTS:
                print(f"  [{label}] … 等待 {RETRY_SLEEP_SEC}s 后重试", flush=True)
                time.sleep(RETRY_SLEEP_SEC)
    raise last_err  # type: ignore[misc]


def _worker_job(payload: dict) -> dict:
    """子进程入口：跑完单个 case，返回 manifest row（须为顶层可 pickle 函数）。"""
    _ensure_paths()
    _load_dotenv()

    index = int(payload["index"])
    label = str(payload["label"])
    case = payload["case"]
    total = int(payload["total"])
    out_root = str(payload["out_root"])
    os.makedirs(out_root, exist_ok=True)

    print("\n" + "#" * 72, flush=True)
    print(f"[{index}/{total}] START {label} case_index={case.get('index')}", flush=True)
    print(f"out: {out_root}", flush=True)
    print("#" * 72, flush=True)

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
        print(
            f"  [OK] {label} best={row.get('best_score')} "
            f"step={row.get('best_step')} ({elapsed}s)",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        row.update({
            "status": "error",
            "error": str(exc),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  [FAIL] {label}: {exc}", flush=True)
        traceback.print_exc()
    return row


def _write_dashboard(session_dir: Path, manifest: dict) -> Path:
    from promptopt.report_trace import write_session_dashboard

    return Path(write_session_dashboard(session_dir, manifest))


def _save_manifest(session_dir: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = session_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _write_dashboard(session_dir, manifest)


# 只认此 run id 及之后的产物（目录名时间戳，不受 report 重建 mtime 影响）
DONE_SINCE_RUN_ID = "20260723_161954"  # case70 新流水线起


def _run_dir_fresh(path: Path) -> bool:
    """t2i_YYYYMMDD_HHMMSS[_nn] / overnight_YYYYMMDD_HHMMSS 是否 ≥ DONE_SINCE_RUN_ID。"""
    name = path.name
    if name.startswith("t2i_"):
        ts = name[4:19]  # 20260723_161954
        return ts >= DONE_SINCE_RUN_ID
    if name.startswith("overnight_"):
        ts = name[len("overnight_"): len("overnight_") + 15]
        return ts >= DONE_SINCE_RUN_ID
    return False


def collect_done_case_indices(outputs_root: Path | None = None) -> set[int]:
    """扫描新流水线之后已成功产出 report 的 case_index。"""
    root = outputs_root or (_PROJECT_ROOT / "outputs")
    done: set[int] = set()

    for man in root.glob("overnight_*/manifest.json"):
        if not _run_dir_fresh(man.parent):
            continue
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in data.get("runs") or []:
            if row.get("status") != "ok":
                continue
            idx = row.get("case_index")
            out = Path(row.get("out_root") or "")
            report = Path(row.get("report") or (out / "report.html"))
            if not report.is_file():
                continue
            if out.name and not _run_dir_fresh(out):
                continue
            if idx is not None:
                done.add(int(idx))
    for summary in root.glob("t2i_*/summary.json"):
        run_dir = summary.parent
        if not _run_dir_fresh(run_dir):
            continue
        if not (run_dir / "report.html").is_file():
            continue
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        idx = data.get("case_index")
        if idx is not None:
            done.add(int(idx))
    return done


def main() -> None:
    _ensure_paths()
    _load_dotenv()
    from promptopt.cases import pick_random_synth_prompt, pick_synth_prompt_by_index

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = _PROJECT_ROOT / "outputs" / f"overnight_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    lib_path = _PROJECT_ROOT / PROMPT_LIBRARY
    done = collect_done_case_indices() if AUTO_SKIP_DONE else set()
    if done:
        print(f"[配置] AUTO_SKIP_DONE：已完成 case {sorted(done)}")

    jobs: list[tuple[str, str, dict]] = []
    if not SKIP_FIRST_CASE:
        if FIRST_CASE_INDEX not in done:
            jobs.append((
                "full_8r",
                "跑满8轮·case70",
                pick_synth_prompt_by_index(FIRST_CASE_INDEX, path=lib_path),
            ))
        else:
            print(f"[配置] case#{FIRST_CASE_INDEX} 已完成，跳过")
    else:
        print(f"[配置] SKIP_FIRST_CASE=True → 跳过 case#{FIRST_CASE_INDEX}")
    if not RUN_FIRST_ONLY:
        for i, seed in enumerate(BATCH_CASE_SEEDS, 1):
            case = pick_random_synth_prompt(seed=seed, path=lib_path)
            if case["index"] in done:
                print(f"[跳过] batch_{i:02d} case#{case['index']} 已有报告")
                continue
            jobs.append((
                f"batch_{i:02d}",
                f"批量{i}/10",
                case,
            ))
    else:
        print("[配置] RUN_FIRST_ONLY=True → 仅跑第一个 case")
    if not jobs:
        print("无待跑任务：目标 case 均已完成")
        _save_manifest(session_dir, {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "total": 0,
            "completed": 0,
            "parallel_jobs": PARALLEL_JOBS,
            "max_rounds": MAX_ROUNDS,
            "runs": [],
            "note": "all_done",
        })
        return

    workers = max(1, min(PARALLEL_JOBS, len(jobs)))
    manifest: dict = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "total": len(jobs),
        "completed": 0,
        "parallel_jobs": workers,
        "max_rounds": MAX_ROUNDS,
        "runs": [],
    }
    _save_manifest(session_dir, manifest)

    print(f"过夜任务启动 total={len(jobs)} parallel={workers} session={session_dir}")
    print("dashboard 入口: outputs/LATEST_DASHBOARD.html")

    payloads = []
    for idx, (job_id, label, case) in enumerate(jobs, 1):
        # 并行时秒级时间戳会撞车，带 index
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{idx:02d}"
        out_root = str(_PROJECT_ROOT / "outputs" / f"t2i_{run_id}")
        payloads.append({
            "index": idx,
            "label": f"{job_id}:{label}",
            "case": case,
            "total": len(jobs),
            "out_root": out_root,
        })

    t_all = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_job, p): p for p in payloads}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "index": p["index"],
                    "label": p["label"],
                    "case_index": (p.get("case") or {}).get("index"),
                    "out_root": p["out_root"],
                    "rounds": MAX_ROUNDS,
                    "status": "error",
                    "error": f"worker_crash: {exc}",
                }
                print(f"  [FAIL] worker crash {p['label']}: {exc}", flush=True)
            results.append(row)
            # 按 index 排序写入，便于 dashboard
            manifest["runs"] = sorted(results, key=lambda r: int(r.get("index") or 0))
            manifest["completed"] = sum(1 for r in results if r.get("status") == "ok")
            _save_manifest(session_dir, manifest)

    manifest["runs"] = sorted(results, key=lambda r: int(r.get("index") or 0))
    manifest["completed"] = sum(1 for r in results if r.get("status") == "ok")
    manifest["total_elapsed_s"] = round(time.time() - t_all, 1)
    _save_manifest(session_dir, manifest)

    print("\n" + "=" * 72)
    print(
        f"全部结束 ok={manifest['completed']}/{manifest['total']} "
        f"· parallel={workers} · {manifest['total_elapsed_s']}s"
    )
    print(f"索引页: {session_dir / 'dashboard.html'}")
    print(f"固定入口: {_PROJECT_ROOT / 'outputs' / 'LATEST_DASHBOARD.html'}")
    print("=" * 72)


if __name__ == "__main__":
    # Windows spawn 需要
    main()
