#!/usr/bin/env python3
# 【功能描述】临时网页：从已有 run 抽取图片 + 审美分数 + 推理文本，便于人工核对
# 【输入】下方 SAMPLE_RUNS；读取 _work/aesthetic_storage 与对应 png
# 【输出】outputs/AESTHETIC_PREVIEW.html

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from statistics import median

_PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = _PROJECT / "outputs"
OUT_HTML = OUTPUTS / "AESTHETIC_PREVIEW.html"
ASSET_DIR = OUTPUTS / "_aesthetic_preview_assets"

# 优先用当前跑、再补历史有 _work 的 run
SAMPLE_RUNS = [
    OUTPUTS / "t2i_20260723_161954",
    OUTPUTS / "t2i_20260722_135533",
    OUTPUTS / "t2i_20260722_233429",
]
MAX_SAMPLES = 8
MIN_DIM_CONF = 0.833


def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


def _load_score(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else None
    return data if isinstance(data, dict) else None


def _collect_pngs(run_dir: Path) -> list[Path]:
    roots = [run_dir / "_work", run_dir / "rounds", run_dir / "initial", run_dir / "best"]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(sorted(root.rglob("*.png")))
    return out


def _match_image(pngs: list[Path], task_dir: Path, image_id: str) -> Path | None:
    """按 eval_task 时间戳就近匹配同名/邻近 png。"""
    task_mtime = task_dir.stat().st_mtime
    # 优先文件名含 train_/gate_/initial
    cands = [p for p in pngs if p.stat().st_mtime <= task_mtime + 5]
    if not cands:
        cands = pngs
    # 时间最近
    best = min(cands, key=lambda p: abs(p.stat().st_mtime - task_mtime))
    # 若 id 像 run_003，尝试同序号
    if image_id.startswith("run_"):
        idx = image_id.split("_")[-1]
        named = [
            p for p in cands
            if p.stem.endswith(f"_{idx}") or p.stem == image_id
        ]
        if named:
            best = min(named, key=lambda p: abs(p.stat().st_mtime - task_mtime))
    return best


def _dim_scores(result: dict) -> list[tuple[str, float]]:
    raw = ((result.get("tournament_details") or {}).get("raw_dimension_scores") or {})
    rows: list[tuple[str, float]] = []
    for major, subs in raw.items():
        if not isinstance(subs, dict):
            continue
        for sub, sc in subs.items():
            try:
                rows.append((f"{major}/{sub}", float(sc)))
            except (TypeError, ValueError):
                pass
    return rows


def _reasons(result: dict, *, min_conf: float) -> tuple[list[dict], list[dict]]:
    """返回 (neg, pos) 条目，含 source/score/reason。"""
    neg: list[dict] = []
    pos: list[dict] = []
    meta = result.get("ensemble_meta") or {}
    conf_map = meta.get("aesthetic_dimension_confidence") or {}
    dim_scores = _dim_scores(result)
    med = median([s for _, s in dim_scores]) if dim_scores else 0.0

    defect = result.get("defect_details") or {}
    for dim, entries in (defect.get("dimension_reasons") or {}).items():
        for e in entries or []:
            if isinstance(e, dict) and e.get("reason"):
                neg.append({
                    "bucket": "defect",
                    "dim": str(dim),
                    "score": e.get("score"),
                    "reason": e["reason"],
                    "conf": None,
                })

    grouped = (result.get("tournament_details") or {}).get("dimension_reasons") or {}
    for major, subs in grouped.items():
        if not isinstance(subs, dict):
            continue
        for sub, entries in subs.items():
            conf = float(conf_map.get(sub, 0) or 0)
            if conf < min_conf:
                continue
            for e in entries or []:
                if not isinstance(e, dict) or not e.get("reason"):
                    continue
                try:
                    sc = float(e.get("score") or 0)
                except (TypeError, ValueError):
                    sc = 0.0
                item = {
                    "bucket": "aesthetic",
                    "dim": f"{major}/{sub}",
                    "score": sc,
                    "reason": e["reason"],
                    "conf": conf,
                    "slot": e.get("slot"),
                }
                if sc and sc < med:
                    neg.append(item)
                else:
                    pos.append(item)
    return neg, pos


def _gather_samples() -> list[dict]:
    samples: list[dict] = []
    seen_img: set[str] = set()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    for run_dir in SAMPLE_RUNS:
        if not run_dir.is_dir():
            continue
        storage = run_dir / "_work" / "aesthetic_storage"
        if not storage.is_dir():
            continue
        pngs = _collect_pngs(run_dir)
        tasks = sorted(storage.glob("eval_task_*"), key=lambda p: p.name)
        # 均匀抽样：头尾 + 中间
        if len(tasks) > MAX_SAMPLES:
            idxs = sorted({0, len(tasks) // 3, 2 * len(tasks) // 3, len(tasks) - 1})
            # 再补几条
            step = max(1, len(tasks) // MAX_SAMPLES)
            idxs = sorted(set(idxs) | set(range(0, len(tasks), step)))[:MAX_SAMPLES]
            tasks = [tasks[i] for i in idxs]

        for task in tasks:
            score_path = task / "3_final_scores.json"
            result = _load_score(score_path)
            if not result:
                continue
            image_id = str(result.get("id") or "image")
            img = _match_image(pngs, task, image_id)
            if not img or not img.is_file():
                continue
            key = str(img.resolve())
            if key in seen_img:
                continue
            seen_img.add(key)

            asset_name = f"{run_dir.name}_{task.name}_{img.name}"
            dest = ASSET_DIR / asset_name
            shutil.copy2(img, dest)

            neg, pos = _reasons(result, min_conf=MIN_DIM_CONF)
            dims = _dim_scores(result)
            med = median([s for _, s in dims]) if dims else None
            defect = result.get("defect_details") or {}
            meta = result.get("ensemble_meta") or {}

            samples.append({
                "run": run_dir.name,
                "task": task.name,
                "image_id": image_id,
                "src": f"_aesthetic_preview_assets/{asset_name}",
                "orig": str(img.relative_to(OUTPUTS)).replace("\\", "/"),
                "final": result.get("final_score"),
                "base": result.get("base_score"),
                "health": result.get("health_factor"),
                "bonus": result.get("bonus_multiplier"),
                "confidence": meta.get("confidence"),
                "defect_total": defect.get("total_defect_score"),
                "defect_dims": defect.get("dimensions") or {},
                "dim_scores": dims,
                "median": med,
                "neg": neg,
                "pos": pos,
            })
            if len(samples) >= MAX_SAMPLES:
                return samples
    return samples


def _render_reasons(items: list[dict], cls: str) -> str:
    if not items:
        return f'<p class="muted">（无 {cls} 条目）</p>'
    parts = []
    for it in items:
        conf = f' · conf={it["conf"]:.2f}' if it.get("conf") is not None else ""
        sc = it.get("score")
        sc_s = f"score={sc}" if sc is not None else ""
        parts.append(
            f'<div class="reason {cls}">'
            f'<div class="r-meta"><b>{_esc(it.get("dim"))}</b> '
            f'<span>{_esc(sc_s)}{conf}</span></div>'
            f'<div class="r-text">{_esc(it.get("reason"))}</div></div>'
        )
    return "".join(parts)


def main() -> None:
    samples = _gather_samples()
    cards = []
    for i, s in enumerate(samples, 1):
        dim_lines = "".join(
            f"<li>{_esc(n)}: <b>{sc:.2f}</b></li>" for n, sc in sorted(s["dim_scores"], key=lambda x: x[1])
        )
        defect_lines = "".join(
            f"<li>{_esc(k)}: {_esc(v)}</li>"
            for k, v in (s["defect_dims"] or {}).items()
            if k != "总缺陷分"
        )
        med = f'{s["median"]:.2f}' if s["median"] is not None else "—"
        cards.append(f"""
<article class="card">
  <header>
    <h2>#{i} · final <span class="score">{_esc(s['final'])}</span></h2>
    <div class="meta">{_esc(s['run'])} / {_esc(s['task'])} · id={_esc(s['image_id'])}</div>
    <div class="meta">img: {_esc(s['orig'])}</div>
  </header>
  <div class="body">
    <figure><img class="lb-src" src="{_esc(s['src'])}" data-caption="#{i} final={_esc(s['final'])}" loading="lazy"/><figcaption>final={_esc(s['final'])} · conf={_esc(s['confidence'])} · 点击放大</figcaption></figure>
    <div class="cols">
      <section>
        <h3>汇总</h3>
        <ul>
          <li>final_score: <b>{_esc(s['final'])}</b></li>
          <li>base_score: {_esc(s['base'])}</li>
          <li>health_factor: {_esc(s['health'])}</li>
          <li>bonus_multiplier: {_esc(s['bonus'])}</li>
          <li>ensemble_confidence: {_esc(s['confidence'])}</li>
          <li>defect_total: {_esc(s['defect_total'])}</li>
          <li>审美维中位(正负分界): <b>{med}</b></li>
        </ul>
        <h3>缺陷维度分</h3>
        <ul>{defect_lines or '<li class="muted">无</li>'}</ul>
        <h3>审美子维分（低→高）</h3>
        <ul class="dims">{dim_lines or '<li class="muted">无</li>'}</ul>
      </section>
      <section>
        <h3 class="neg-h">负向反馈（缺陷全进 + 审美&lt;中位）</h3>
        {_render_reasons(s['neg'], 'neg')}
      </section>
      <section>
        <h3 class="pos-h">正向反馈（审美≥中位）</h3>
        {_render_reasons(s['pos'], 'pos')}
      </section>
    </div>
  </div>
</article>
""")

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>审美分数与推理预览（临时）</title>
<style>
:root {{ --bg:#0f1115; --card:#171a21; --line:#2a3140; --text:#e8eaed; --muted:#9aa0a6;
  --neg:#f07178; --pos:#3dd68c; --acc:#8ab4f8; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); }}
.wrap {{ max-width:1200px; margin:0 auto; padding:1.2rem 1rem 3rem; }}
h1 {{ font-size:1.25rem; margin:0 0 .4rem }}
.note {{ color:#fdd663; font-size:.85rem; margin-bottom:1.2rem; line-height:1.45 }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; margin:0 0 1.2rem; overflow:hidden }}
.card header {{ padding:.8rem 1rem; border-bottom:1px solid var(--line) }}
.card h2 {{ margin:0; font-size:1.05rem }}
.score {{ color:var(--acc) }}
.meta {{ color:var(--muted); font-size:.75rem; margin-top:.2rem; word-break:break-all }}
.body {{ padding:1rem }}
figure {{ margin:0 0 1rem }}
figure img {{ width:100%; max-height:420px; object-fit:contain; background:#0a0c10; border-radius:8px; border:1px solid var(--line); cursor:zoom-in }}
figcaption {{ color:var(--muted); font-size:.75rem; margin-top:.3rem }}
.lb-overlay{{display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.88);align-items:center;justify-content:center;flex-direction:column;gap:.6rem}}
.lb-overlay.open{{display:flex}}
.lb-overlay img{{max-width:min(96vw,1600px);max-height:86vh;object-fit:contain;border-radius:6px}}
.lb-cap{{color:#c5cad6;font-size:.85rem}}
.lb-nav{{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:64px;border:0;border-radius:8px;background:rgba(255,255,255,.08);color:#fff;font-size:1.6rem;cursor:pointer}}
.lb-prev{{left:12px}}.lb-next{{right:12px}}
.lb-close{{position:absolute;top:12px;right:16px;border:0;background:transparent;color:#fff;font-size:1.8rem;cursor:pointer}}
.lb-hint{{position:absolute;bottom:12px;color:#8b93a7;font-size:.72rem}}
.cols {{ display:grid; grid-template-columns:1fr 1.2fr 1.2fr; gap:.8rem }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:1fr }} }}
section h3 {{ margin:.2rem 0 .5rem; font-size:.85rem }}
.neg-h {{ color:var(--neg) }} .pos-h {{ color:var(--pos) }}
ul {{ margin:.2rem 0 .8rem 1.1rem; padding:0; font-size:.8rem; line-height:1.45 }}
.dims li {{ font-variant-numeric:tabular-nums }}
.reason {{ border:1px solid var(--line); border-radius:6px; padding:.45rem .55rem; margin:0 0 .4rem; font-size:.78rem; line-height:1.45 }}
.reason.neg {{ border-left:3px solid var(--neg); background:rgba(240,113,120,.06) }}
.reason.pos {{ border-left:3px solid var(--pos); background:rgba(61,214,140,.06) }}
.r-meta {{ color:var(--muted); margin-bottom:.2rem; display:flex; justify-content:space-between; gap:.5rem; flex-wrap:wrap }}
.r-text {{ color:var(--text) }}
.muted {{ color:var(--muted) }}
</style>
</head>
<body>
<div class="wrap">
  <h1>审美分数与推理预览（临时）</h1>
  <p class="note">
    从已有 run 的 aesthetic_storage 抽取 {len(samples)} 张图。<br/>
    负向 = 全部缺陷 reason + 审美子维 reason（score &lt; 本图中位）；正向 = 审美 reason（score ≥ 中位）。
    仅展示 dim_confidence ≥ {MIN_DIM_CONF} 的审美推理。用于核对「分数/文案是否合适」，非正式报告。
  </p>
  {''.join(cards) if cards else '<p class="muted">未找到带评分的样本（_work 可能已清理）</p>'}
</div>
<div class="lb-overlay" id="lb">
  <button type="button" class="lb-close" id="lbClose">&times;</button>
  <button type="button" class="lb-nav lb-prev" id="lbPrev">&#8249;</button>
  <img id="lbImg" src="" alt=""/>
  <button type="button" class="lb-nav lb-next" id="lbNext">&#8250;</button>
  <div class="lb-cap" id="lbCap"></div>
  <div class="lb-hint">← → 切换 · Esc 关闭</div>
</div>
<script>
(function(){{
  var imgs = Array.prototype.slice.call(document.querySelectorAll('img.lb-src'));
  if (!imgs.length) return;
  var ov = document.getElementById('lb');
  var el = document.getElementById('lbImg');
  var cap = document.getElementById('lbCap');
  var idx = 0;
  function show(i){{
    idx = (i + imgs.length) % imgs.length;
    var n = imgs[idx];
    el.src = n.currentSrc || n.src;
    cap.textContent = (n.getAttribute('data-caption') || '') + ' (' + (idx+1) + '/' + imgs.length + ')';
    ov.classList.add('open');
  }}
  function hide(){{ ov.classList.remove('open'); el.src = ''; }}
  imgs.forEach(function(img, i){{ img.addEventListener('click', function(){{ show(i); }}); }});
  document.getElementById('lbClose').onclick = hide;
  document.getElementById('lbPrev').onclick = function(e){{ e.stopPropagation(); show(idx-1); }};
  document.getElementById('lbNext').onclick = function(e){{ e.stopPropagation(); show(idx+1); }};
  ov.onclick = function(e){{ if (e.target === ov) hide(); }};
  document.addEventListener('keydown', function(e){{
    if (!ov.classList.contains('open')) return;
    if (e.key === 'Escape') hide();
    else if (e.key === 'ArrowLeft') show(idx-1);
    else if (e.key === 'ArrowRight') show(idx+1);
  }});
}})();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"written {OUT_HTML}")
    print(f"samples={len(samples)}")
    print(f"http://127.0.0.1:9788/AESTHETIC_PREVIEW.html")


if __name__ == "__main__":
    main()
