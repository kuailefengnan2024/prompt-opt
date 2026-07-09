# 【功能描述】T2I 训练产物 HTML 报告（横向时间线 + 全量 diff + meta + 纵向组图）
# 【输入】outputs/t2i_<ts>/
# 【输出】report.html

from __future__ import annotations

import difflib
import html
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _rel(root: Path, path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return ""


def _esc(text: str) -> str:
    return html.escape(text or "")


def _fmt_score(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        return f"{f * 100:.1f}" if f <= 1.0 else f"{f:.1f}"
    except (TypeError, ValueError):
        return str(v)


def _render_full_diff(old: str, new: str) -> str:
    """全量 diff：相等段完整展示，删除红、新增绿。"""
    old = old or ""
    new = new or ""
    if old == new:
        return '<span class="diff-same">（本轮无文本变动）</span>'

    sm = difflib.SequenceMatcher(None, old, new)
    chunks: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            seg = old[i1:i2]
            if seg:
                chunks.append(f'<span class="diff-same">{_esc(seg)}</span>')
        elif tag == "delete":
            chunks.append(f'<del class="diff-del">{_esc(old[i1:i2])}</del>')
        elif tag == "insert":
            chunks.append(f'<ins class="diff-add">{_esc(new[j1:j2])}</ins>')
        elif tag == "replace":
            chunks.append(f'<del class="diff-del">{_esc(old[i1:i2])}</del>')
            chunks.append(f'<ins class="diff-add">{_esc(new[j1:j2])}</ins>')
    return "".join(chunks) or '<span class="diff-same">（无差异）</span>'


def _render_image_stack(
    root: Path,
    folder: Path,
    prefix: str,
    scores: list[dict] | None,
    group_label: str,
) -> str:
    if not folder.is_dir():
        return ""
    score_map = {str(s.get("id", "")): s for s in (scores or []) if isinstance(s, dict)}
    rows: list[str] = []
    for img in sorted(folder.glob(f"{prefix}_*.png")):
        rel = _rel(root, img)
        if not rel:
            continue
        sid = img.stem
        sc = score_map.get(sid, {})
        final = sc.get("final_score")
        if final is None and sc.get("soft") is not None:
            final = float(sc["soft"]) * 100
        rows.append(
            f'<figure class="img-row">'
            f'<img src="{_esc(rel)}" alt="{_esc(sid)}" loading="lazy"/>'
            f'<figcaption>{_esc(group_label)} · {_esc(sid)} · <b>{_fmt_score(final)}</b></figcaption>'
            f"</figure>"
        )
    if not rows:
        return ""
    return f'<div class="img-group"><div class="img-group-label">{_esc(group_label)}</div>{"".join(rows)}</div>'


def _extract_meta(result: dict, meta_file: dict | None) -> str:
    if isinstance(meta_file, dict) and meta_file.get("meta_prompt_content"):
        return str(meta_file["meta_prompt_content"]).strip()
    mp = result.get("meta_prompt")
    if isinstance(mp, dict) and mp.get("meta_prompt_content"):
        return str(mp["meta_prompt_content"]).strip()
    return ""


def _action_cls(action: str) -> str:
    a = (action or "").lower()
    if "accept" in a:
        return "accept"
    if "reject" in a:
        return "reject"
    return "neutral"


def _collect_rounds(root: Path) -> list[dict[str, Any]]:
    rounds_dir = root / "rounds"
    if not rounds_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for rd in sorted(rounds_dir.glob("round_*")):
        if not rd.is_dir():
            continue
        m = re.search(r"round_(\d+)", rd.name)
        idx = int(m.group(1)) if m else 0
        result = _read_json(rd / "result.json") or {}
        rollout_scores = _read_json(rd / "rollout" / "scores.json")
        gate_scores = _read_json(rd / "gate" / "scores.json")
        meta_file = _read_json(rd / "meta_prompt.json")
        out.append({
            "idx": idx,
            "dir": rd,
            "result": result,
            "input_prompt": _read_text(rd / "input_prompt.md"),
            "candidate_prompt": _read_text(rd / "candidate_prompt.md"),
            "rollout_scores": rollout_scores,
            "gate_scores": gate_scores,
            "meta_text": _extract_meta(result, meta_file),
        })
    return sorted(out, key=lambda x: x["idx"])


def _round_card(r: dict[str, Any], root: Path, is_best: bool) -> str:
    res = r["result"]
    rd: Path = r["dir"]
    action = str(res.get("action", ""))
    cls = _action_cls(action)
    best_mark = ' <span class="best-star">★</span>' if is_best else ""
    diff_html = _render_full_diff(r["input_prompt"], r["candidate_prompt"])
    gate_sc = _fmt_score(res.get("gate_soft"))
    rollout_sc = _fmt_score(res.get("rollout_soft"))
    short_action = "✓" if "accept" in action.lower() else "✗" if "reject" in action.lower() else "·"

    rollout_imgs = _render_image_stack(root, rd / "rollout", "train", r.get("rollout_scores"), "Rollout")
    gate_imgs = _render_image_stack(root, rd / "gate", "gate", r.get("gate_scores"), "Gate")
    imgs = rollout_imgs + gate_imgs
    if not imgs:
        imgs = '<div class="img-empty">无图片</div>'

    meta_html = (
        f'<div class="rc-meta"><div class="block-label">Meta 记忆</div><div class="meta-body">{_esc(r["meta_text"])}</div></div>'
        if r["meta_text"]
        else '<div class="rc-meta empty"><div class="block-label">Meta 记忆</div><span class="muted">（空）</span></div>'
    )

    return f"""
<article class="round-card {cls}">
  <header class="rc-head">
    <span class="rc-num">R{r['idx']}</span>
    <span class="rc-action">{_esc(short_action)}</span>
    <span class="rc-sub">roll {rollout_sc} → gate {gate_sc}</span>{best_mark}
  </header>
  <div class="rc-body">
    <div class="rc-imgs">{imgs}</div>
    <div class="rc-diff"><div class="block-label">Prompt 改动</div><div class="diff-body">{diff_html}</div></div>
    {meta_html}
  </div>
</article>"""


def generate_run_report(out_root: str | Path) -> str:
    root = Path(out_root).resolve()
    summary = _read_json(root / "summary.json") or {}
    case = _read_json(root / "case.json") or {}
    meta_final = _read_json(root / "meta_prompt.json") or {}

    initial_score = _read_json(root / "initial" / "score.json") or {}
    best_score = _read_json(root / "best" / "score.json") or {}
    initial_img = _rel(root, root / "initial" / "image.png")
    best_img = _rel(root, root / "best" / "image.png")
    init_sc = _fmt_score(initial_score.get("final_score") or summary.get("initial", {}).get("final_score"))
    best_sc = _fmt_score(best_score.get("best_final_score") or summary.get("best", {}).get("final_score"))
    best_step = best_score.get("best_step") or summary.get("best", {}).get("step") or "—"

    meta_final_text = str(
        meta_final.get("meta_prompt_content")
        or summary.get("meta_prompt", {}).get("final_content")
        or ""
    ).strip()

    rounds = _collect_rounds(root)
    run_name = root.name
    case_index = summary.get("case_index")
    if case_index is None:
        case_index = case.get("index")
    title = f"case #{case_index}" if case_index is not None else run_name
    case_label = f"index={case_index}" if case_index is not None else "—"

    round_cards = "".join(_round_card(r, root, is_best=(r["idx"] == best_step)) for r in rounds)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>T2I · {_esc(title)}</title>
<style>
:root {{
  --bg:#0c0e14; --panel:#141820; --border:#262c3a;
  --text:#e6e9f0; --muted:#7d8498;
  --del-bg:rgba(240,90,90,.25); --del-fg:#ff9a9a;
  --add-bg:rgba(60,200,120,.25); --add-fg:#7dffb0;
  --accept:#3dd68c; --reject:#f07178; --accent:#6ea8fe;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex;flex-direction:column}}
.top{{flex:0 0 auto;padding:.6rem 1rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:1rem;background:var(--panel);flex-wrap:wrap}}
.top h1{{font-size:.95rem;font-weight:600}}
.top .meta{{font-size:.75rem;color:var(--muted)}}
.top .chip{{font-size:.72rem;padding:.15rem .5rem;border-radius:5px;background:#1c2130;border:1px solid var(--border)}}
.top .chip.up{{color:var(--accept);border-color:rgba(61,214,140,.4)}}
.layout{{flex:1;display:flex;min-height:0}}
.anchor{{flex:0 0 240px;border-right:1px solid var(--border);display:flex;flex-direction:column;background:var(--panel);padding:.5rem;gap:.5rem;overflow-y:auto}}
.anchor-card{{flex:0 0 auto;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:#10141c}}
.anchor-card h2{{font-size:.68rem;color:var(--muted);padding:.35rem .5rem;border-bottom:1px solid var(--border)}}
.anchor-card .body{{padding:.35rem}}
.anchor-card img{{width:100%;border-radius:4px;display:block}}
.anchor-card .sc{{text-align:center;font-size:.8rem;font-weight:700;padding:.3rem}}
.anchor-card .sc.init{{color:var(--muted)}}
.anchor-card .sc.best{{color:var(--accept)}}
.anchor-meta{{flex:1;min-height:120px;display:flex;flex-direction:column}}
.anchor-meta .meta-scroll{{flex:1;overflow:auto;padding:.45rem .5rem;font-size:.65rem;line-height:1.5;color:#b8bfd0;max-height:40vh}}
.timeline-wrap{{flex:1;min-width:0;display:flex;flex-direction:column}}
.timeline{{flex:1;overflow:auto;padding:.5rem;display:flex;align-items:flex-start;gap:.5rem}}
.round-card{{flex:0 0 260px;border:1px solid var(--border);border-radius:8px;background:var(--panel);overflow:hidden}}
.round-card.accept{{border-color:rgba(61,214,140,.5)}}
.round-card.reject{{border-color:rgba(240,113,120,.4)}}
.rc-head{{display:flex;align-items:center;gap:.35rem;padding:.35rem .5rem;border-bottom:1px solid var(--border);font-size:.72rem;flex-wrap:wrap}}
.rc-num{{font-weight:700;color:var(--accent)}}
.rc-action{{font-weight:700}}
.round-card.accept .rc-action{{color:var(--accept)}}
.round-card.reject .rc-action{{color:var(--reject)}}
.rc-sub{{color:var(--muted);font-size:.65rem}}
.best-star{{color:#ffd166}}
.rc-body{{display:flex;flex-direction:column}}
.block-label{{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.25rem}}
.img-group{{border-bottom:1px solid var(--border)}}
.img-group-label{{font-size:.62rem;color:var(--accent);padding:.25rem .45rem;background:#10141c;border-bottom:1px solid var(--border)}}
.img-row{{margin:0;padding:.3rem .4rem;border-bottom:1px solid #1e2430}}
.img-row:last-child{{border-bottom:none}}
.img-row img{{width:100%;display:block;border-radius:4px;border:1px solid var(--border)}}
.img-row figcaption{{font-size:.6rem;color:var(--muted);margin-top:.2rem}}
.img-empty{{padding:.5rem;font-size:.7rem;color:var(--muted)}}
.rc-diff{{padding:.4rem .45rem;border-bottom:1px solid var(--border);background:#10141c}}
.diff-body{{font-size:.66rem;line-height:1.6;white-space:pre-wrap;word-break:break-word}}
.diff-del{{background:var(--del-bg);color:var(--del-fg);text-decoration:line-through}}
.diff-add{{background:var(--add-bg);color:var(--add-fg);text-decoration:none}}
.diff-same{{color:#8b92a8}}
.rc-meta{{padding:.4rem .45rem;background:#0f1218}}
.rc-meta.empty{{color:var(--muted);font-size:.65rem}}
.meta-body{{font-size:.64rem;line-height:1.55;white-space:pre-wrap;word-break:break-word;color:#b0b8cc}}
.legend{{flex:0 0 auto;padding:.3rem 1rem;border-top:1px solid var(--border);font-size:.65rem;color:var(--muted);display:flex;gap:.8rem;flex-wrap:wrap;background:var(--panel)}}
.legend del{{background:var(--del-bg);color:var(--del-fg);text-decoration:line-through;padding:0 2px}}
.legend ins{{background:var(--add-bg);color:var(--add-fg);text-decoration:none;padding:0 2px}}
.muted{{color:var(--muted)}}
</style>
</head>
<body>
<div class="top">
  <h1>{_esc(title)}</h1>
  <span class="meta">{_esc(case_label)} · {_esc(run_name)}</span>
  <span class="chip">Initial {init_sc}</span>
  <span class="chip up">Best {best_sc} @R{best_step}</span>
</div>
<div class="layout">
  <aside class="anchor">
    <div class="anchor-card">
      <h2>输入 · Initial</h2>
      <div class="body">{"<img src='" + _esc(initial_img) + "' alt='initial'/>" if initial_img else "<span class='muted'>无图</span>"}</div>
      <div class="sc init">{init_sc}</div>
    </div>
    <div class="anchor-card">
      <h2>输出 · Best</h2>
      <div class="body">{"<img src='" + _esc(best_img) + "' alt='best'/>" if best_img else "<span class='muted'>无图</span>"}</div>
      <div class="sc best">{best_sc}</div>
    </div>
    <div class="anchor-card anchor-meta">
      <h2>最终 Meta 记忆</h2>
      <div class="meta-scroll">{_esc(meta_final_text) if meta_final_text else "（无）"}</div>
    </div>
  </aside>
  <div class="timeline-wrap">
    <section class="timeline">{round_cards}</section>
  </div>
</div>
<div class="legend">
  <span><del>改前</del> <ins>改后</ins></span>
  <span>每卡上图 Rollout（当前 prompt）下图 Gate（候选 prompt），一图一行</span>
</div>
</body>
</html>"""

    out_path = root / "report.html"
    out_path.write_text(page, encoding="utf-8")
    return str(out_path)
