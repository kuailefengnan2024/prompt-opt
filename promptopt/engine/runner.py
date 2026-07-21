# 【功能描述】T2I 单任务 Reflect 优化主循环（精简产物：initial/ + best/）
# 【输入】initial_prompt、runtime 参数、out_root、可选 case_meta
# 【输出】outputs 下 case.json、initial/、best/、summary.json

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptopt.clients.aesthetic_bridge import build_aesthetic_client, configure_aesthetic_storage, score_image_sync
from promptopt.design_requirement import resolve_design_requirement
from promptopt.clients.api_core_bridge import configure_api_clients, generate_image_sync
from promptopt.trajectory import build_prediction_entry, format_rollout_trajectory
from promptopt.evaluation.gate import evaluate_gate
from promptopt.gradient.aggregate import merge_patches
from promptopt.gradient.reflect import run_minibatch_reflect
from promptopt.model.api_core_backend import configure_api_core_llm, get_token_summary, reset_token_tracker
from promptopt.model.backend_config import set_optimizer_backend
from promptopt.optimizer.meta_prompt import build_history_digest, run_meta_prompt_update
from promptopt.optimizer.prompt_editor import apply_patch_with_report
from promptopt.report import generate_run_report
from promptopt.utils import compute_score


@dataclass
class T2IRunConfig:
    initial_prompt: str
    category: str
    max_rounds: int
    train_runs: int
    gate_runs: int
    edit_budget: int
    hard_threshold: float
    min_ensemble_confidence: float
    reason_min_dim_confidence: float
    llm_provider: str
    image_provider: str
    image_size: str
    aesthetic_ensemble: dict[str, str]
    minibatch_size: int
    analyst_workers: int
    merge_batch_size: int
    seed: int
    out_root: str
    case_meta: dict[str, Any] | None = None
    save_debug: bool = False
    use_meta_prompt: bool = True
    meta_prompt_max_chars: int = 1500

    @property
    def design_requirement(self) -> str:
        """审美打分与优化链路的结构化设计要求锚点。"""
        return resolve_design_requirement(self.initial_prompt, self.case_meta)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _rollout_score_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "id": r.get("id"),
            "soft": r.get("soft"),
            "final_score": r.get("final_score"),
            "hard": r.get("hard"),
            "image": (r.get("extras") or {}).get("image_path"),
        })
    return rows


def _write_templates_manifest(out_root: str) -> None:
    from promptopt.templates import list_prompts

    _write_json(os.path.join(out_root, "templates.json"), {
        "note": "全部模板在 promptopt/prompts/，{占位符} 由 templates.fill_prompt 填充",
        "prompt_dir": "promptopt/prompts/",
        "prompts": {name: f"promptopt/prompts/{name}.md" for name in list_prompts()},
        "fill_helper": "promptopt/templates.py::fill_prompt",
        "update": "promptopt/optimizer/prompt_editor.py（纯代码 apply）",
        "gate": "promptopt/evaluation/gate.py（纯分数比较）",
    })


def _mirror_images(src_dir: str, dest_dir: str, label: str) -> None:
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        if not name.startswith(f"{label}_") or not name.endswith(".png"):
            continue
        shutil.copy2(os.path.join(src_dir, name), os.path.join(dest_dir, name))


def _mirror_patches(patches_dir: str, dest_dir: str) -> None:
    if not os.path.isdir(patches_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)
    for name in sorted(os.listdir(patches_dir)):
        if not name.endswith(".json"):
            continue
        shutil.copy2(os.path.join(patches_dir, name), os.path.join(dest_dir, name))


def _persist_round_artifacts(
    *,
    out_root: str,
    round_idx: int,
    input_prompt: str,
    rollout_results: list[dict[str, Any]],
    rollout_dir: str,
    patches_dir: str | None = None,
    merged_patch: dict[str, Any] | None = None,
    candidate_prompt: str | None = None,
    gate_results: list[dict[str, Any]] | None = None,
    gate_dir: str | None = None,
    step_rec: dict[str, Any] | None = None,
) -> None:
    rd = Path(out_root) / "rounds" / f"round_{round_idx:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "input_prompt.md").write_text(input_prompt, encoding="utf-8")
    _write_json(str(rd / "rollout" / "scores.json"), _rollout_score_rows(rollout_results))
    _mirror_images(rollout_dir, str(rd / "rollout"), "train")
    if patches_dir:
        _mirror_patches(patches_dir, str(rd / "patch"))
    if merged_patch is not None:
        _write_json(str(rd / "patch" / "merged_patch.json"), merged_patch)
    if candidate_prompt is not None:
        (rd / "candidate_prompt.md").write_text(candidate_prompt, encoding="utf-8")
    if gate_results is not None:
        _write_json(str(rd / "gate" / "scores.json"), _rollout_score_rows(gate_results))
    if gate_dir:
        _mirror_images(gate_dir, str(rd / "gate"), "gate")
    if step_rec is not None:
        _write_json(str(rd / "result.json"), step_rec)
    meta_snapshot = step_rec.get("meta_prompt") if isinstance(step_rec, dict) else None
    if meta_snapshot is not None:
        _write_json(str(rd / "meta_prompt.json"), meta_snapshot)


def _update_meta_prompt(
    *,
    cfg: T2IRunConfig,
    out_root: str,
    meta_prompt_content: str,
    history: list[dict[str, Any]],
    round_idx: int,
    prompt_excerpt: str,
) -> tuple[str, dict[str, Any] | None]:
    """滚动更新跨轮记忆，返回 (新记忆, 本轮 meta 快照)。"""
    if not cfg.use_meta_prompt:
        return meta_prompt_content, None

    digest = build_history_digest(history)
    t0 = time.time()
    result = run_meta_prompt_update(
        prompt_excerpt=prompt_excerpt,
        design_requirement=cfg.design_requirement,
        previous_meta=meta_prompt_content,
        round_digest=digest,
        max_chars=cfg.meta_prompt_max_chars,
    )
    elapsed = round(time.time() - t0, 1)
    snapshot: dict[str, Any] = {
        "round": round_idx,
        "previous": meta_prompt_content,
        "digest_chars": len(digest),
        "time_s": elapsed,
    }
    if result and result.get("meta_prompt_content"):
        new_content = str(result["meta_prompt_content"]).strip()
        snapshot.update({
            "action": "updated",
            "reasoning": result.get("reasoning", ""),
            "meta_prompt_content": new_content,
        })
        _write_json(os.path.join(out_root, "meta_prompt.json"), {
            "round": round_idx,
            "meta_prompt_content": new_content,
            "reasoning": result.get("reasoning", ""),
            "updated_at_round": round_idx,
        })
        print(f"  [Meta] 记忆已更新 ({len(new_content)} chars, {elapsed}s)")
        return new_content, snapshot

    snapshot["action"] = "unchanged"
    snapshot["meta_prompt_content"] = meta_prompt_content
    print(f"  [Meta] 记忆未变 ({elapsed}s)")
    return meta_prompt_content, snapshot


def _pick_best_rollout(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [r for r in results if r.get("soft") is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r.get("soft", 0) or 0))


def _score_to_rollout(
    *,
    run_idx: int,
    prompt: str,
    image_bytes: bytes,
    image_path: str,
    design_requirement: str,
    category: str,
    hard_threshold: float,
    reason_min_dim_confidence: float,
    min_ensemble_confidence: float,
) -> dict[str, Any]:
    score_data, err = score_image_sync(
        image_bytes=image_bytes,
        design_requirement=design_requirement,
        category=category,
        image_name=f"run_{run_idx:03d}",
    )
    if err or not score_data:
        return {
            "id": f"run_{run_idx:03d}",
            "hard": 0,
            "soft": 0.0,
            "fail_reason": f"aesthetic_error: {err}",
            "task_type": "t2i",
            "extras": {},
        }

    meta = score_data.get("ensemble_meta") or {}
    ensemble_conf = float(meta.get("confidence", 0) or 0)
    final_score = float(score_data.get("final_score", 0) or 0)
    soft = final_score / 100.0
    hard = 1 if final_score >= hard_threshold else 0

    traj = format_rollout_trajectory(
        run_idx=run_idx,
        prompt_excerpt=prompt,
        image_path=image_path,
        score_result=score_data,
        hard=hard,
        soft=soft,
        min_dim_confidence=reason_min_dim_confidence,
    )

    fail_reason = ""
    if hard == 0:
        fail_reason = f"low_aesthetic: final_score={final_score:.2f}"
    if ensemble_conf < min_ensemble_confidence:
        fail_reason = (fail_reason + "; " if fail_reason else "") + f"low_confidence={ensemble_conf:.3f}"

    return {
        "id": f"run_{run_idx:03d}",
        "hard": hard,
        "soft": soft,
        "fail_reason": fail_reason or "ok",
        "task_type": "t2i",
        "final_score": final_score,
        "ensemble_confidence": ensemble_conf,
        "aesthetic_detail": score_data,
        "trajectory_text": traj,
        "extras": {"image_path": image_path},
    }


def _rollout_prompt(
    *,
    prompt: str,
    cfg: T2IRunConfig,
    out_dir: str,
    num_runs: int,
    label: str,
) -> list[dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    predictions_dir = os.path.join(out_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    results: list[dict[str, Any]] = []

    for i in range(num_runs):
        run_id = f"{label}_{i:03d}"
        img_path = os.path.join(out_dir, f"{run_id}.png")
        print(f"      [{label}] 生图 {i + 1}/{num_runs} ...")
        image_bytes, gen_err = generate_image_sync(prompt, size=cfg.image_size)
        if gen_err or not image_bytes:
            results.append({
                "id": run_id,
                "hard": 0,
                "soft": 0.0,
                "fail_reason": f"image_gen_error: {gen_err}",
                "task_type": "t2i",
            })
            continue
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        print(f"      [{label}] 审美打分 {i + 1}/{num_runs} ...")
        row = _score_to_rollout(
            run_idx=i,
            prompt=prompt,
            image_bytes=image_bytes,
            image_path=img_path,
            design_requirement=cfg.design_requirement,
            category=cfg.category,
            hard_threshold=cfg.hard_threshold,
            reason_min_dim_confidence=cfg.reason_min_dim_confidence,
            min_ensemble_confidence=cfg.min_ensemble_confidence,
        )
        row["id"] = run_id
        pred = build_prediction_entry(run_idx=i, trajectory_text=row.get("trajectory_text", ""))
        pred["id"] = run_id
        pred_dir = os.path.join(predictions_dir, run_id)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(pred["conversation"], f, ensure_ascii=False, indent=2)
        results.append(row)
        print(
            f"      [{label}] run {i + 1}: soft={row.get('soft', 0):.4f} "
            f"final={row.get('final_score', 0)} hard={row.get('hard', 0)}"
        )
    return results


def _generate_preview_image(prompt: str, cfg: T2IRunConfig, dest: str) -> dict[str, Any] | None:
    print("  生成预览图 ...")
    image_bytes, gen_err = generate_image_sync(prompt, size=cfg.image_size)
    if gen_err or not image_bytes:
        print(f"  预览图生成失败: {gen_err}")
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(image_bytes)
    row = _score_to_rollout(
        run_idx=0,
        prompt=prompt,
        image_bytes=image_bytes,
        image_path=dest,
        design_requirement=cfg.design_requirement,
        category=cfg.category,
        hard_threshold=cfg.hard_threshold,
        reason_min_dim_confidence=cfg.reason_min_dim_confidence,
        min_ensemble_confidence=cfg.min_ensemble_confidence,
    )
    print(f"  预览审美: final={row.get('final_score', 0)} soft={row.get('soft', 0):.4f}")
    return row


def _copy_image(src: str | None, dest: str) -> bool:
    if not src or not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _slim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """summary 用轻量历史，避免 merged_patch / meta_prompt 撑大 JSON。"""
    slim: list[dict[str, Any]] = []
    for rec in history:
        if not isinstance(rec, dict):
            continue
        row = {k: v for k, v in rec.items() if k not in ("merged_patch", "meta_prompt")}
        slim.append(row)
    return slim


def _write_deliverables(
    *,
    out_root: str,
    cfg: T2IRunConfig,
    prompt_v0: str,
    best_prompt: str,
    best_score: float,
    best_step: int,
    initial_image: str | None,
    best_image: str | None,
    initial_score_row: dict[str, Any] | None,
    history: list[dict[str, Any]],
    meta_prompt_content: str = "",
) -> dict[str, Any]:
    root = Path(out_root)
    initial_dir = root / "initial"
    best_dir = root / "best"
    initial_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    (initial_dir / "prompt.md").write_text(prompt_v0, encoding="utf-8")
    (best_dir / "prompt.md").write_text(best_prompt, encoding="utf-8")

    if initial_image:
        _copy_image(initial_image, str(initial_dir / "image.png"))
    if best_image:
        _copy_image(best_image, str(best_dir / "image.png"))
    elif initial_image and best_step == 0:
        _copy_image(initial_image, str(best_dir / "image.png"))

    best_final = best_score * 100.0 if best_score >= 0 else None
    score_payload = {
        "best_score": best_score,
        "best_final_score": best_final,
        "best_step": best_step,
    }
    if initial_score_row:
        score_payload["initial_final_score"] = initial_score_row.get("final_score")
        score_payload["initial_soft"] = initial_score_row.get("soft")
    (best_dir / "score.json").write_text(
        json.dumps(score_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (root / "initial_prompt.txt").write_text(cfg.initial_prompt.strip(), encoding="utf-8")
    if cfg.case_meta:
        with open(root / "case.json", "w", encoding="utf-8") as f:
            json.dump(cfg.case_meta, f, ensure_ascii=False, indent=2)

    summary = {
        "case_index": (cfg.case_meta or {}).get("index"),
        "case_source": (cfg.case_meta or {}).get("source"),
        "initial_prompt_chars": len(cfg.initial_prompt),
        "initial": {
            "prompt": "initial/prompt.md",
            "image": "initial/image.png" if (initial_dir / "image.png").is_file() else None,
            "final_score": (initial_score_row or {}).get("final_score"),
        },
        "best": {
            "prompt": "best/prompt.md",
            "image": "best/image.png" if (best_dir / "image.png").is_file() else None,
            "score": best_score,
            "final_score": best_final,
            "step": best_step,
        },
        "rounds": cfg.max_rounds,
        "history": _slim_history(history),
        "meta_prompt": {
            "enabled": cfg.use_meta_prompt,
            "final_content": meta_prompt_content or None,
            "chars": len(meta_prompt_content) if meta_prompt_content else 0,
        },
        "report": "report.html",
        "token_summary": get_token_summary(),
    }
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    config_payload = {k: v for k, v in cfg.__dict__.items() if k not in ("case_meta", "initial_prompt")}
    config_payload["initial_prompt_chars"] = len(cfg.initial_prompt)
    if cfg.case_meta:
        config_payload["case_index"] = cfg.case_meta.get("index")
        config_payload["case_source"] = cfg.case_meta.get("source")
    with open(root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    return summary


def run_t2i_optimize(cfg: T2IRunConfig) -> dict[str, Any]:
    out_root = cfg.out_root
    work_root = os.path.join(out_root, "_work")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(work_root, exist_ok=True)
    _write_templates_manifest(out_root)

    aesthetic_storage = os.path.join(work_root, "aesthetic_storage")
    configure_aesthetic_storage(aesthetic_storage)
    configure_api_clients(llm_provider=cfg.llm_provider, image_provider=cfg.image_provider)
    configure_api_core_llm(llm_provider=cfg.llm_provider, image_provider=cfg.image_provider)
    set_optimizer_backend("api_core")
    build_aesthetic_client(cfg.aesthetic_ensemble)
    reset_token_tracker()

    prompt_v0 = (cfg.initial_prompt or "").strip()
    if not prompt_v0:
        raise ValueError("initial_prompt 为空")
    design_req = cfg.design_requirement
    print(f"\n[Init] 使用库内 prompt（{len(prompt_v0)} chars），跳过 Phase0 合成")
    print(f"  设计要求: {len(design_req)} chars")
    if cfg.case_meta and cfg.case_meta.get("index") is not None:
        print(f"  case_index={cfg.case_meta.get('index')}")

    initial_dir = Path(out_root) / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    (initial_dir / "prompt.md").write_text(prompt_v0, encoding="utf-8")

    initial_preview = os.path.join(work_root, "initial_preview.png")
    initial_score_row = _generate_preview_image(prompt_v0, cfg, initial_preview)
    if os.path.isfile(initial_preview):
        shutil.copy2(initial_preview, str(initial_dir / "image.png"))
    if initial_score_row:
        _write_json(str(initial_dir / "score.json"), {
            "final_score": initial_score_row.get("final_score"),
            "soft": initial_score_row.get("soft"),
            "image": "initial/image.png",
        })

    current_prompt = prompt_v0
    best_prompt = prompt_v0
    best_score = float(initial_score_row.get("soft", -1.0)) if initial_score_row else -1.0
    current_score = best_score
    best_step = 0
    best_image_path: str | None = initial_preview if initial_score_row else None
    history: list[dict[str, Any]] = []
    meta_prompt_content = ""

    for round_idx in range(1, cfg.max_rounds + 1):
        step_dir = os.path.join(work_root, f"round_{round_idx:03d}")
        os.makedirs(step_dir, exist_ok=True)
        t0 = time.time()
        print(f"\n{'=' * 60}\n[Round {round_idx}/{cfg.max_rounds}]\n{'=' * 60}")
        if meta_prompt_content:
            print(f"  [Meta] 注入记忆 {len(meta_prompt_content)} chars")
        round_input_prompt = current_prompt

        rollout_dir = os.path.join(step_dir, "rollout")
        rollout_results = _rollout_prompt(
            prompt=current_prompt,
            cfg=cfg,
            out_dir=rollout_dir,
            num_runs=cfg.train_runs,
            label="train",
        )
        _, r_soft = compute_score(rollout_results)
        print(f"  [1/5 Rollout] soft={r_soft:.4f}")

        patches_dir = os.path.join(step_dir, "patches")
        raw_patches = run_minibatch_reflect(
            rollout_results,
            current_prompt,
            os.path.join(rollout_dir, "predictions"),
            patches_dir,
            cfg.analyst_workers,
            False,
            minibatch_size=cfg.minibatch_size,
            edit_budget=cfg.edit_budget,
            random_seed=cfg.seed + round_idx,
            error_system=None,
            success_system=None,
            update_mode="patch",
            meta_prompt_context=meta_prompt_content,
            design_requirement=design_req,
        )
        failure_patches = []
        success_patches = []
        for p in raw_patches:
            if not isinstance(p, dict):
                continue
            inner = p.get("patch", p)
            if not isinstance(inner, dict):
                continue
            edits = inner.get("edits") or []
            if not edits:
                continue
            if p.get("source_type") == "success":
                success_patches.append(inner)
            else:
                failure_patches.append(inner)

        if not failure_patches and not success_patches:
            print("  [skip] 无可用 patch")
            step_rec = {
                "round": round_idx,
                "action": "skip_no_patches",
                "rollout_soft": r_soft,
                "current_score": current_score,
                "best_score": best_score,
            }
            history.append(step_rec)
            meta_prompt_content, meta_snap = _update_meta_prompt(
                cfg=cfg,
                out_root=out_root,
                meta_prompt_content=meta_prompt_content,
                history=history,
                round_idx=round_idx,
                prompt_excerpt=prompt_v0,
            )
            if meta_snap:
                step_rec["meta_prompt"] = meta_snap
            _persist_round_artifacts(
                out_root=out_root,
                round_idx=round_idx,
                input_prompt=current_prompt,
                rollout_results=rollout_results,
                rollout_dir=rollout_dir,
                patches_dir=patches_dir,
                step_rec={"round": round_idx, "action": "skip_no_patches", "rollout_soft": r_soft, "meta_prompt": meta_snap},
            )
            continue

        merged_patch = merge_patches(
            current_prompt,
            failure_patches,
            success_patches,
            batch_size=cfg.merge_batch_size,
            verbose=True,
            workers=cfg.analyst_workers,
            update_mode="patch",
            meta_prompt_context=meta_prompt_content,
            design_requirement=design_req,
            edit_budget=cfg.edit_budget,
        )
        candidate_prompt, _apply_report = apply_patch_with_report(current_prompt, merged_patch)

        gate_dir = os.path.join(step_dir, "gate_rollout")
        gate_results = _rollout_prompt(
            prompt=candidate_prompt,
            cfg=cfg,
            out_dir=gate_dir,
            num_runs=cfg.gate_runs,
            label="gate",
        )
        cand_hard, cand_soft = compute_score(gate_results)
        gate = evaluate_gate(
            candidate_prompt=candidate_prompt,
            cand_hard=cand_soft,
            current_prompt=current_prompt,
            current_score=current_score,
            best_prompt=best_prompt,
            best_score=best_score,
            best_step=best_step,
            global_step=round_idx,
        )
        current_prompt = gate.current_prompt
        current_score = gate.current_score
        best_prompt = gate.best_prompt
        best_score = gate.best_score
        best_step = gate.best_step

        if gate.action == "accept_new_best":
            best_row = _pick_best_rollout(gate_results)
            if best_row:
                best_image_path = (best_row.get("extras") or {}).get("image_path")

        step_rec = {
            "round": round_idx,
            "action": gate.action,
            "rollout_soft": r_soft,
            "gate_soft": cand_soft,
            "current_score": current_score,
            "best_score": best_score,
            "best_step": best_step,
            "wall_time_s": round(time.time() - t0, 1),
            "merged_patch": merged_patch,
        }
        history.append(step_rec)
        print(
            f"  [5/5 Gate] action={gate.action} gate_soft={cand_soft:.4f} "
            f"current={current_score:.4f} best={best_score:.4f}"
        )
        meta_prompt_content, meta_snap = _update_meta_prompt(
            cfg=cfg,
            out_root=out_root,
            meta_prompt_content=meta_prompt_content,
            history=history,
            round_idx=round_idx,
            prompt_excerpt=prompt_v0,
        )
        if meta_snap:
            step_rec["meta_prompt"] = meta_snap
        _persist_round_artifacts(
            out_root=out_root,
            round_idx=round_idx,
            input_prompt=round_input_prompt,
            rollout_results=rollout_results,
            rollout_dir=rollout_dir,
            patches_dir=patches_dir,
            merged_patch=merged_patch,
            candidate_prompt=candidate_prompt,
            gate_results=gate_results,
            gate_dir=gate_dir,
            step_rec=step_rec,
        )

    summary = _write_deliverables(
        out_root=out_root,
        cfg=cfg,
        prompt_v0=prompt_v0,
        best_prompt=best_prompt,
        best_score=best_score,
        best_step=best_step,
        initial_image=initial_preview if os.path.isfile(initial_preview) else None,
        best_image=best_image_path,
        initial_score_row=initial_score_row,
        history=history,
        meta_prompt_content=meta_prompt_content,
    )

    if not cfg.save_debug:
        shutil.rmtree(work_root, ignore_errors=True)

    report_path = generate_run_report(out_root)
    summary["report"] = report_path
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
