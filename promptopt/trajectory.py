# 【功能描述】将审美打分结果格式化为 Reflect trajectory；正/负反馈分栏供 LLM 做 keep/fix
# 【输入】ComprehensiveResult dict、图片路径、prompt 片段、置信阈值
# 【输出】含 positive_feedback / negative_feedback 分区的轨迹字符串

from __future__ import annotations

from statistics import median
from typing import Any


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj if isinstance(obj, dict) else {}


def _collect_dim_scores(result: dict[str, Any]) -> list[tuple[str, float]]:
    """扁平化审美子维度分数 [(major/sub, score), ...]。"""
    tournament = _as_dict(result.get("tournament_details"))
    raw = tournament.get("raw_dimension_scores") or {}
    out: list[tuple[str, float]] = []
    for major, subs in raw.items():
        if not isinstance(subs, dict):
            continue
        for sub, score in subs.items():
            try:
                out.append((f"{major}/{sub}", float(score)))
            except (TypeError, ValueError):
                continue
    return out


def _collect_aesthetic_reasons(
    result: dict[str, Any],
    *,
    min_dim_confidence: float,
) -> list[tuple[str, str, float, float]]:
    """返回 (sub, reason, score, conf)。"""
    meta = _as_dict(result.get("ensemble_meta"))
    dim_conf_map = meta.get("aesthetic_dimension_confidence") or {}
    tournament = _as_dict(result.get("tournament_details"))
    grouped = tournament.get("dimension_reasons") or {}
    rows: list[tuple[str, str, float, float]] = []
    for _major, subs in grouped.items():
        if not isinstance(subs, dict):
            continue
        for sub, entries in subs.items():
            conf = float(dim_conf_map.get(sub, 0) or 0)
            if conf < min_dim_confidence:
                continue
            for entry in entries or []:
                if not isinstance(entry, dict) or not entry.get("reason"):
                    continue
                try:
                    sc = float(entry.get("score") or 0)
                except (TypeError, ValueError):
                    sc = 0.0
                rows.append((str(sub), str(entry["reason"]), sc, conf))
    return rows


def format_feedback_sections(
    result: dict[str, Any],
    *,
    min_dim_confidence: float,
) -> tuple[str, str]:
    """拆出负向 / 正向反馈文本。程序只分栏，keep/fix 由 LLM 判断。"""
    neg: list[str] = []
    pos: list[str] = []

    meta = _as_dict(result.get("ensemble_meta"))
    overall_conf = float(meta.get("confidence", 0) or 0)
    header = f"ensemble_confidence={overall_conf:.3f}"

    defect = _as_dict(result.get("defect_details"))
    dims = _as_dict(defect.get("dimensions"))
    if dims:
        neg.append("缺陷维度分:")
        for key, val in dims.items():
            if key == "总缺陷分":
                continue
            neg.append(f"  - {key}: {val}")

    for dim, entries in (defect.get("dimension_reasons") or {}).items():
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("reason"):
                neg.append(
                    f"  [缺陷·{dim}] {entry.get('reason')} (score={entry.get('score')})"
                )

    dim_scores = _collect_dim_scores(result)
    if dim_scores:
        med = median([s for _, s in dim_scores])
        neg.append(f"审美子维度（低于中位 {med:.2f}）:")
        pos.append(f"审美子维度（不低于中位 {med:.2f}）:")
        for name, sc in sorted(dim_scores, key=lambda x: x[1]):
            line = f"  - {name}: {sc:.2f}"
            if sc < med:
                neg.append(line)
            else:
                pos.append(line)

        reasons = _collect_aesthetic_reasons(result, min_dim_confidence=min_dim_confidence)
        for sub, reason, sc, conf in reasons:
            line = f"  [审美·{sub}] {reason} (score={sc}, dim_conf={conf:.3f})"
            # reason 自带 score 时按该分相对中位分栏；无则可两边都给 LLM 看的保守策略：跟维度分一致
            if sc and sc < med:
                neg.append(line)
            else:
                pos.append(line)

    neg_body = "\n".join(neg) if len(neg) > 1 else "（无明显负向条目，仅供参考总分）"
    pos_body = "\n".join(pos) if len(pos) > 1 else "（无明显正向条目，仅供参考总分）"
    return (
        f"{header}\n\n## 负向反馈（优先考虑修改相关构图描述）\n{neg_body}",
        f"## 正向反馈（优先保留相关构图描述）\n{pos_body}",
    )


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
    neg, pos = format_feedback_sections(
        score_result,
        min_dim_confidence=min_dim_confidence,
    )
    rel = "success_relative" if hard else "failure_relative"
    return (
        f"### Rollout #{run_idx}\n"
        f"label={rel} hard={hard} soft={soft:.4f} final_score={final_score}\n"
        f"image={image_path}\n"
        f"prompt_excerpt={prompt_excerpt[:400]}\n"
        f"feedback:\n{neg}\n\n{pos}\n"
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
