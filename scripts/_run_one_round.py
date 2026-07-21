#!/usr/bin/env python3
# 临时：跑 1 轮优化并输出 report 路径
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_AESTHETIC = Path(r"D:\kv-generator\engines\aesthetic-core")
if _AESTHETIC.exists() and str(_AESTHETIC) not in sys.path:
    sys.path.insert(0, str(_AESTHETIC))


def _load_dotenv() -> None:
    for path in (_ROOT / ".env", Path(r"D:\api-core\.env"), Path(r"D:\kv-generator\.env")):
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


def main() -> None:
    _load_dotenv()
    from promptopt.cases import pick_random_synth_prompt
    from promptopt.engine.runner import T2IRunConfig, run_t2i_optimize

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = str(_ROOT / "outputs" / f"t2i_{run_id}")
    case = pick_random_synth_prompt(seed=48, path=_ROOT / "data/kv_synth_prompts_100_only.json")
    print(f"case_index={case['index']} chars={len(case['prompt'])}")
    print(f"out_root={out_root}")

    cfg = T2IRunConfig(
        initial_prompt=case["prompt"],
        category="3d",
        max_rounds=1,
        train_runs=4,
        gate_runs=2,
        edit_budget=5,
        hard_threshold=65.0,
        min_ensemble_confidence=0.5,
        reason_min_dim_confidence=0.833,
        llm_provider="doubao21pro",
        image_provider="seedream_4_5",
        image_size="2999x1687",
        aesthetic_ensemble={"doubao": "doubao_seed_2_0_pro_vision"},
        minibatch_size=8,
        analyst_workers=4,
        merge_batch_size=4,
        seed=42,
        out_root=out_root,
        case_meta=case,
        save_debug=False,
        use_meta_prompt=True,
        meta_prompt_max_chars=1500,
    )
    summary = run_t2i_optimize(cfg)
    report = summary.get("report") or str(Path(out_root) / "report.html")
    best = summary.get("best") or {}
    print(f"REPORT_PATH={report}")
    print(f"best_score={best.get('score')}")
    print(f"best_step={best.get('step')}")


if __name__ == "__main__":
    main()
