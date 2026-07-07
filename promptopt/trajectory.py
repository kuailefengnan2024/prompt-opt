# 【功能描述】将审美打分结果格式化为 Reflect trajectory 文本
# 【输入】ComprehensiveResult dict、图片路径、prompt 片段
# 【输出】analyst 可读的轨迹字符串与高置信 reason 摘要

from __future__ import annotations

from typing import Any


def _collect_high_confidence_reasons(
    result: dict[str, Any],
    *,
    min_dim_confidence: float,
) -> list[str]:
    lines: list[str] = []
    meta = result.get("ensemble_meta") or {}
    overall_conf = float(meta.get("confidence", 0) or 0)
    lines.append(f"ensemble_confidence={overall_conf:.3f}")

    defect = result.get("defect_details") or {}
    if hasattr(defect, "model_dump"):
        defect = defect.model_dump()
    dims = defect.get("dimensions") or {}
    if hasattr(dims, "model_dump"):
        dims = dims.model_dump()
    if dims:
        lines.append("缺陷维度:")
        for key, val in dims.items():
            if key == "总缺陷分":
                continue
            lines.append(f"  - {key}: {val}")

    defect_reasons = defect.get("dimension_reasons") or {}
    for dim, entries in defect_reasons.items():
        for entry in entries or []:
            if isinstance(entry, dict):
                lines.append(f"  [缺陷·{dim}] {entry.get('reason', '')} (score={entry.get('score')})")

    tournament = result.get("tournament_details") or {}
    if hasattr(tournament, "model_dump"):
        tournament = tournament.model_dump()
    raw_dims = tournament.get("raw_dimension_scores") or {}
    if raw_dims:
        lines.append("审美子维度均分:")
        for major, subs in raw_dims.items():
            if isinstance(subs, dict):
                for sub, score in subs.items():
                    lines.append(f"  - {major}/{sub}: {score}")

    dim_conf_map = meta.get("aesthetic_dimension_confidence") or {}
    grouped = tournament.get("dimension_reasons") or {}
    for major, subs in grouped.items():
        if not isinstance(subs, dict):
            continue
        for sub, entries in subs.items():
            conf = float(dim_conf_map.get(sub, 0) or 0)
            if conf < min_dim_confidence:
                continue
            for entry in entries or []:
                if isinstance(entry, dict) and entry.get("reason"):
                    lines.append(
                        f"  [审美·{sub}] {entry['reason']} "
                        f"(score={entry.get('score')}, dim_conf={conf:.3f})"
                    )
    return lines


def format_rollout_trajectory(
    *,
    run_idx: int,
    prompt_excerpt: str,
    image_path: str,
    score_result: dict[str, Any],
    hard: int,
    soft: float,
    min_dim_confidence: float,
) -> str:
    final_score = score_result.get("final_score", soft * 100)
    reason_lines = _collect_high_confidence_reasons(
        score_result,
        min_dim_confidence=min_dim_confidence,
    )
    body = "\n".join(reason_lines) if reason_lines else "(无高置信 reason，仅参考分数)"
    return (
        f"### Rollout #{run_idx}\n"
        f"hard={hard} soft={soft:.4f} final_score={final_score}\n"
        f"image={image_path}\n"
        f"prompt_excerpt={prompt_excerpt[:400]}\n"
        f"feedback:\n{body}"
    )


def build_prediction_entry(
    *,
    run_idx: int,
    trajectory_text: str,
) -> dict[str, Any]:
    return {
        "id": f"run_{run_idx:03d}",
        "conversation": [
            {"type": "tool_call", "cmd": "t2i_generate_and_score", "obs": trajectory_text},
        ],
    }
