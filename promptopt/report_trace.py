# 【功能描述】训练产物可视化：横向轮次流水线、prompt diff、LLM 注入分区；多任务 Tab 仪表盘
# 【输入】单 run 的 out_root（rounds/initial/best/…）；或 overnight manifest
# 【输出】report.html；可选 LATEST_DASHBOARD.html（头部任务切换 Tab）

from __future__ import annotations

import difflib
import html
import json
import re
from pathlib import Path
from typing import Any

try:
    import markdown as md_lib
except ImportError:  # pragma: no cover
    md_lib = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MD = md_lib.Markdown(extensions=["tables", "fenced_code", "sane_lists", "nl2br"]) if md_lib else None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        return f"{f * 100:.1f}" if f <= 1.0 else f"{f:.1f}"
    except (TypeError, ValueError):
        return str(v)


def _md(text: object) -> str:
    raw = ("" if text is None else str(text)).strip()
    if not raw:
        return '<p class="muted">（空）</p>'
    lines: list[str] = []
    for line in raw.splitlines():
        m = re.fullmatch(r"(【[^】]+】)\s*", line.strip())
        if m:
            lines.append(f"### {m.group(1)}")
        else:
            lines.append(line)
    prepared = "\n".join(lines)
    if _MD is None:
        return f'<pre class="sec-body mono">{_esc(prepared)}</pre>'
    _MD.reset()
    return f'<div class="md-body">{_MD.convert(prepared)}</div>'


def _diff_html(old: str, new: str) -> str:
    old, new = old or "", new or ""
    if not new:
        return '<span class="diff-same">（本轮无 candidate / skip）</span>'
    if old == new:
        return '<span class="diff-same">（本轮文本无变动）</span>'
    chunks: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if tag == "equal":
            if old[i1:i2]:
                chunks.append(f'<span class="diff-same">{_esc(old[i1:i2])}</span>')
        elif tag == "delete":
            chunks.append(f'<del class="diff-del">{_esc(old[i1:i2])}</del>')
        elif tag == "insert":
            chunks.append(f'<ins class="diff-add">{_esc(new[j1:j2])}</ins>')
        elif tag == "replace":
            chunks.append(f'<del class="diff-del">{_esc(old[i1:i2])}</del>')
            chunks.append(f'<ins class="diff-add">{_esc(new[j1:j2])}</ins>')
    return f'<div class="diff-body">{"".join(chunks)}</div>'


def _section(tag: str, title: str, body: str, note: str = "", *, as_md: bool = True) -> str:
    body = body or ""
    note_html = f'<div class="sec-note">{_esc(note)}</div>' if note else ""
    content = _md(body) if as_md else f'<pre class="sec-body mono">{_esc(body)}</pre>'
    return (
        f'<div class="llm-sec tag-{_esc(tag)}">'
        f'<div class="sec-head"><span class="sec-tag">{_esc(tag)}</span>'
        f'<span class="sec-title">{_esc(title)}</span>'
        f'<span class="sec-len">{len(body)} 字</span></div>'
        f"{note_html}"
        f'<div class="sec-body">{content}</div></div>'
    )


def _img(run_dir: Path, rel: str, cap: str) -> str:
    if not (run_dir / rel).is_file():
        return ""
    src = _esc(rel.replace("\\", "/"))
    return (
        f'<figure class="thumb">'
        f'<img class="lb-src" src="{src}" alt="{_esc(cap)}" loading="lazy" '
        f'data-caption="{_esc(cap)}"/>'
        f"<figcaption>{_esc(cap)}</figcaption></figure>"
    )


LIGHTBOX_CSS = r"""
.thumb img.lb-src{cursor:zoom-in}
.lb-overlay{display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.88);align-items:center;justify-content:center;flex-direction:column;gap:.6rem}
.lb-overlay.open{display:flex}
.lb-overlay img{max-width:min(96vw,1600px);max-height:86vh;object-fit:contain;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,.5)}
.lb-cap{color:#c5cad6;font-size:.85rem;max-width:90vw;text-align:center}
.lb-nav{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:64px;border:0;border-radius:8px;background:rgba(255,255,255,.08);color:#fff;font-size:1.6rem;cursor:pointer;line-height:1}
.lb-nav:hover{background:rgba(255,255,255,.18)}
.lb-prev{left:12px}.lb-next{right:12px}
.lb-close{position:absolute;top:12px;right:16px;border:0;background:transparent;color:#fff;font-size:1.8rem;cursor:pointer;line-height:1;padding:.2rem .5rem}
.lb-hint{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);color:#8b93a7;font-size:.72rem}
"""


LIGHTBOX_HTML = """
<div class="lb-overlay" id="lb" role="dialog" aria-modal="true">
  <button type="button" class="lb-close" id="lbClose" aria-label="close">&times;</button>
  <button type="button" class="lb-nav lb-prev" id="lbPrev" aria-label="prev">&#8249;</button>
  <img id="lbImg" src="" alt=""/>
  <button type="button" class="lb-nav lb-next" id="lbNext" aria-label="next">&#8250;</button>
  <div class="lb-cap" id="lbCap"></div>
  <div class="lb-hint">← → 切换 · Esc 关闭 · 点击背景关闭</div>
</div>
"""


LIGHTBOX_JS = r"""
<script>
(function(){
  var imgs = Array.prototype.slice.call(document.querySelectorAll('img.lb-src'));
  if (!imgs.length) return;
  var ov = document.getElementById('lb');
  var el = document.getElementById('lbImg');
  var cap = document.getElementById('lbCap');
  var idx = 0;
  function show(i){
    if (!imgs.length) return;
    idx = (i + imgs.length) % imgs.length;
    var n = imgs[idx];
    el.src = n.currentSrc || n.src;
    cap.textContent = (n.getAttribute('data-caption') || n.alt || '') + '  (' + (idx+1) + '/' + imgs.length + ')';
    ov.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
  function hide(){
    ov.classList.remove('open');
    el.src = '';
    document.body.style.overflow = '';
  }
  imgs.forEach(function(img, i){
    img.addEventListener('click', function(e){ e.preventDefault(); e.stopPropagation(); show(i); });
  });
  document.getElementById('lbClose').addEventListener('click', hide);
  document.getElementById('lbPrev').addEventListener('click', function(e){ e.stopPropagation(); show(idx-1); });
  document.getElementById('lbNext').addEventListener('click', function(e){ e.stopPropagation(); show(idx+1); });
  ov.addEventListener('click', function(e){ if (e.target === ov) hide(); });
  document.addEventListener('keydown', function(e){
    if (!ov.classList.contains('open')) return;
    if (e.key === 'Escape') hide();
    else if (e.key === 'ArrowLeft') show(idx-1);
    else if (e.key === 'ArrowRight') show(idx+1);
  });
})();
</script>
"""


def _traj_from_scores(scores: list) -> str:
    parts = []
    for row in scores or []:
        t = str(row.get("trajectory_text") or "").strip()
        if t:
            parts.append(t)
    if parts:
        return "\n\n".join(parts)
    return (
        "【未落盘】本轮 scores.json 无 trajectory_text；"
        "新跑会落盘正/负向反馈。\n"
    )


def _edits_summary(patch: dict | None) -> str:
    if not patch:
        return '<p class="muted">（无 patch）</p>'
    edits = patch.get("edits") or []
    rows = []
    for i, e in enumerate(edits, 1):
        if not isinstance(e, dict):
            continue
        rows.append(
            f"<div class='edit-card'>"
            f"<div class='edit-op'>{_esc(e.get('op'))} #{i}</div>"
            f"<div class='edit-label'>target（删/换）</div>"
            f"<div class='edit-t'>{_md(e.get('target', ''))}</div>"
            f"<div class='edit-label'>content（增）</div>"
            f"<div class='edit-c'>{_md(e.get('content', ''))}</div>"
            f"</div>"
        )
    reasoning = patch.get("reasoning") or ""
    head = f"<div class='edit-reason'>{_md(reasoning)}</div>" if reasoning else ""
    return head + ("".join(rows) or '<p class="muted">（空 edits）</p>')


CSS = r"""
:root{
  --bg:#0b0d12; --panel:#12161f; --panel2:#181d28; --line:#2a3142;
  --text:#e8eaed; --muted:#8b93a7; --acc:#7eb6ff;
  --del-bg:rgba(240,90,90,.28); --del:#ff9a9a;
  --add-bg:rgba(60,200,120,.28); --add:#7dffb0;
  --tag-scope:#c4a35a; --tag-design:#c084fc; --tag-meta:#60a5fa;
  --tag-prompt:#34d399; --tag-traj:#f87171; --tag-budget:#fbbf24;
  --tag-patch:#a78bfa; --tag-other:#94a3b8;
  --ok:#3dd68c; --bad:#f07178; --skip:#9aa0a6;
}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}
.top{flex:0 0 auto;padding:.7rem 1.1rem;border-bottom:1px solid var(--line);background:var(--panel);display:flex;flex-wrap:wrap;gap:.8rem;align-items:center}
.top h1{margin:0;font-size:1rem}
.top .meta{color:var(--muted);font-size:.78rem}
.chip{font-size:.72rem;padding:.15rem .5rem;border-radius:999px;border:1px solid var(--line);background:#1a2030}
.chip.up{color:var(--ok);border-color:rgba(61,214,140,.4)}
.legend{display:flex;gap:.7rem;flex-wrap:wrap;font-size:.7rem;color:var(--muted)}
.legend del{background:var(--del-bg);color:var(--del);text-decoration:line-through;padding:0 3px}
.legend ins{background:var(--add-bg);color:var(--add);text-decoration:none;padding:0 3px}
.task-tabs{flex:0 0 auto;display:flex;gap:.35rem;padding:.45rem 1rem;border-bottom:1px solid var(--line);background:#0e1219;overflow-x:auto}
.task-tab{flex:0 0 auto;border:1px solid var(--line);background:#151a24;color:var(--muted);padding:.35rem .7rem;border-radius:6px;cursor:pointer;font-size:.75rem}
.task-tab.active{color:var(--text);border-color:var(--acc);background:#1a2740}
.task-tab .sc{color:var(--ok);margin-left:.35rem}
.board{flex:1;min-height:0;overflow-x:auto;overflow-y:hidden;display:flex;gap:0;padding:0}
.col{flex:0 0 520px;border-right:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0}
.col.init{flex-basis:360px;background:var(--panel2)}
.col-head{flex:0 0 auto;padding:.65rem .75rem;border-bottom:1px solid var(--line);position:sticky;top:0;background:inherit;z-index:2}
.col-head h2{margin:0;font-size:.95rem;display:flex;align-items:center;gap:.4rem}
.col-head .scores{font-size:.72rem;color:var(--muted);margin-top:.25rem}
.badge{font-size:.68rem;font-weight:700;padding:.1rem .4rem;border-radius:4px}
.badge.accept,.badge.accept_new_best{background:rgba(61,214,140,.15);color:var(--ok)}
.badge.reject{background:rgba(240,113,120,.15);color:var(--bad)}
.badge.skip_no_patches{background:#2a2f3a;color:var(--skip)}
.col-body{flex:1;min-height:0;overflow-y:auto;padding:.65rem .75rem 1.4rem}
.block{margin:0 0 .75rem}
.block-label{font-size:.65rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem}
.thumbs{display:flex;flex-wrap:wrap;gap:.35rem}
.thumb{margin:0;width:110px}
.thumb img{width:100%;display:block;border-radius:5px;border:1px solid var(--line)}
.thumb figcaption{font-size:.62rem;color:var(--muted);margin-top:.15rem}
""" + LIGHTBOX_CSS + r"""
.diff-body{font-size:.68rem;line-height:1.55;white-space:pre-wrap;word-break:break-word;background:#0e1219;border:1px solid var(--line);border-radius:8px;padding:.55rem;max-height:280px;overflow:auto}
.diff-del{background:var(--del-bg);color:var(--del);text-decoration:line-through}
.diff-add{background:var(--add-bg);color:var(--add);text-decoration:none}
.diff-same{color:#8b92a8}
.step{border:1px solid var(--line);border-radius:10px;margin:0 0 .7rem;overflow:hidden;background:#0f131b}
.step-head{display:flex;align-items:center;gap:.45rem;padding:.4rem .55rem;background:#171c27;border-bottom:1px solid var(--line);font-size:.72rem;font-weight:650}
.step-num{flex:0 0 auto;width:1.35rem;height:1.35rem;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#243049;color:var(--acc);font-size:.68rem;font-weight:700}
.step-eng .step-head{background:#1a2234;border-bottom-color:#2e3b55}
.step-eng{border-color:#334666}
.step-body{padding:.55rem}
.llm-sec{border:1px solid var(--line);border-radius:8px;margin:0 0 .45rem;overflow:hidden;background:#0e1219}
.sec-head{display:flex;align-items:center;gap:.4rem;padding:.3rem .5rem;background:#151a24;border-bottom:1px solid var(--line);font-size:.7rem}
.sec-tag{font-weight:700;font-size:.62rem;padding:.08rem .35rem;border-radius:4px;color:#0b0d12}
.tag-scope .sec-tag{background:var(--tag-scope)}
.tag-design .sec-tag{background:var(--tag-design)}
.tag-meta .sec-tag{background:var(--tag-meta)}
.tag-prompt .sec-tag{background:var(--tag-prompt)}
.tag-traj .sec-tag{background:var(--tag-traj)}
.tag-budget .sec-tag{background:var(--tag-budget)}
.tag-patch .sec-tag{background:var(--tag-patch)}
.tag-other .sec-tag{background:var(--tag-other)}
.tag-scope{border-left:3px solid var(--tag-scope)}
.tag-design{border-left:3px solid var(--tag-design)}
.tag-meta{border-left:3px solid var(--tag-meta)}
.tag-prompt{border-left:3px solid var(--tag-prompt)}
.tag-traj{border-left:3px solid var(--tag-traj)}
.tag-budget{border-left:3px solid var(--tag-budget)}
.tag-patch{border-left:3px solid var(--tag-patch)}
.sec-title{color:var(--text)}
.sec-len{margin-left:auto;color:var(--muted);font-size:.62rem}
.sec-note{font-size:.65rem;color:#fdd663;padding:.25rem .5rem;background:#1a1810}
.sec-body{margin:0;padding:.45rem .55rem;font-size:.72rem;line-height:1.55;max-height:260px;overflow:auto;color:#c5cad6}
.sec-body.mono,.sec-body pre{white-space:pre-wrap;word-break:break-word;font-size:.66rem;font-family:ui-monospace,Consolas,monospace}
.md-body{color:#d5dae6}
.md-body>*:first-child{margin-top:0}
.md-body>*:last-child{margin-bottom:0}
.md-body h1,.md-body h2,.md-body h3,.md-body h4{margin:.55rem 0 .3rem;color:#f0f3f8;font-weight:650;line-height:1.3}
.md-body h1{font-size:1rem}.md-body h2{font-size:.92rem}.md-body h3{font-size:.84rem}.md-body h4{font-size:.78rem}
.md-body p{margin:.35rem 0}
.md-body ul,.md-body ol{margin:.3rem 0 .3rem 1.1rem;padding:0}
.md-body li{margin:.15rem 0}
.md-body table{border-collapse:collapse;width:100%;margin:.4rem 0;font-size:.68rem}
.md-body th,.md-body td{border:1px solid var(--line);padding:.25rem .4rem;text-align:left;vertical-align:top}
.md-body th{background:#1a2030;color:#e8eaed}
.md-body code{background:#1e2433;padding:.05rem .25rem;border-radius:3px;font-size:.68rem}
.md-body pre{background:#0a0d13;border:1px solid var(--line);border-radius:6px;padding:.45rem;overflow:auto}
.md-body pre code{background:transparent;padding:0}
.md-body strong{color:#fff;font-weight:650}
.md-body hr{border:none;border-top:1px solid var(--line);margin:.5rem 0}
.md-body blockquote{margin:.35rem 0;padding:.2rem .6rem;border-left:3px solid var(--acc);color:var(--muted)}
.edit-card{border:1px solid var(--line);border-radius:8px;padding:.45rem;margin:.35rem 0;background:#0e1219}
.edit-op{font-size:.72rem;font-weight:700;color:var(--acc);margin-bottom:.25rem}
.edit-label{font-size:.62rem;color:var(--muted);margin:.25rem 0 .1rem}
.edit-t{margin:0;padding:.35rem;border-radius:4px;background:var(--del-bg);max-height:140px;overflow:auto}
.edit-t .md-body,.edit-t .md-body p,.edit-t .md-body h3{color:var(--del)!important}
.edit-c{margin:0;padding:.35rem;border-radius:4px;background:var(--add-bg);max-height:180px;overflow:auto}
.edit-c .md-body,.edit-c .md-body p,.edit-c .md-body h3{color:var(--add)!important}
.edit-reason{font-size:.72rem;color:#c5cad6;margin-bottom:.4rem;line-height:1.45}
.note{font-size:.7rem;color:#fdd663;background:#1a1810;border:1px solid #4a3d18;border-radius:6px;padding:.4rem .5rem;margin-bottom:.6rem}
.muted{color:var(--muted);font-size:.72rem}
.frame-wrap{flex:1;min-height:0;border:0}
.frame-wrap iframe{width:100%;height:100%;border:0;background:var(--bg)}
"""


def _step(num: str, title: str, body: str, *, eng: bool = False) -> str:
    cls = "step step-eng" if eng else "step"
    tag = "工程 · LLM" if eng else "执行"
    return (
        f'<div class="{cls}">'
        f'<div class="step-head"><span class="step-num">{_esc(num)}</span>'
        f'<span>{_esc(title)}</span>'
        f'<span class="muted" style="margin-left:auto">{tag}</span></div>'
        f'<div class="step-body">{body}</div></div>'
    )


def _resolve_design(run_dir: Path, initial_prompt: str, case: dict) -> str:
    from promptopt.design_requirement import format_design_requirement_section, resolve_design_requirement

    saved = _read(run_dir / "design_requirement.txt").strip()
    if saved:
        design = saved
    else:
        design = resolve_design_requirement(
            initial_prompt,
            case,
            design_requirement=str(case.get("design_requirement") or ""),
        )
    return format_design_requirement_section(design).strip()


def build_run_report_html(run_dir: str | Path, *, embed_tabs: str = "") -> str:
    """生成单任务全流程 HTML 正文（含可选头部任务 Tab）。"""
    from promptopt.optimizer.meta_prompt import format_meta_prompt_context
    from promptopt.templates import clear_cache, get_prompt_scope_section

    clear_cache()
    root = Path(run_dir).resolve()
    case = _read_json(root / "case.json") or {}
    cfg = _read_json(root / "config.json") or {}
    summary = _read_json(root / "summary.json") or {}
    initial_prompt = _read(root / "initial" / "prompt.md") or _read(root / "initial_prompt.txt")
    design_section = _resolve_design(root, initial_prompt, case)
    scope = get_prompt_scope_section()
    edit_budget = cfg.get("edit_budget", 5)
    use_rel = bool(cfg.get("use_relative_success", True))
    hard_th = float(cfg.get("hard_threshold", 65))
    best_step = (summary.get("best") or {}).get("step")

    cols: list[str] = []
    cols.append(f"""
<div class="col init">
  <div class="col-head">
    <h2>Init</h2>
    <div class="scores">baseline · {_esc(_fmt((summary.get('initial') or {}).get('final_score')))}</div>
  </div>
  <div class="col-body">
    <div class="note">横向滚动查看各轮。红=删/旧 · 绿=增/新。每轮步骤 5 标出 prompt 改动。</div>
    <div class="block"><div class="block-label">Initial 图</div>
      <div class="thumbs">{_img(root, 'initial/image.png', f"init {_fmt((summary.get('initial') or {}).get('final_score'))}")}</div>
    </div>
    <div class="block"><div class="block-label">设计要求（约束锚点）</div>
      <div class="sec-body" style="max-height:160px">{_md(design_section or '（空）')}</div>
    </div>
    <div class="block"><div class="block-label">case / config</div>
      {_md(
        "- case: **#" + str(case.get("index")) + "**\n"
        f"- edit_budget: **{edit_budget}**\n"
        f"- 分流: **{'相对均值' if use_rel else f'hard≥{hard_th}'}**\n"
        f"- train/gate: **{cfg.get('train_runs')}** / **{cfg.get('gate_runs')}**"
      )}
    </div>
    <div class="block"><div class="block-label">初始 prompt（渲染）</div>
      <div class="sec-body" style="max-height:320px">{_md(initial_prompt)}</div>
    </div>
  </div>
</div>
""")

    rounds_dir = root / "rounds"
    round_dirs = sorted(rounds_dir.glob("round_*")) if rounds_dir.is_dir() else []
    for rd in round_dirs:
        ridx = int(rd.name.split("_")[-1])
        result = _read_json(rd / "result.json") or {}
        action = str(result.get("action") or "?")
        badge = action if action in ("accept", "accept_new_best", "reject", "skip_no_patches") else "skip_no_patches"
        input_prompt = _read(rd / "input_prompt.md")
        candidate = _read(rd / "candidate_prompt.md")
        rollout_scores = _read_json(rd / "rollout" / "scores.json") or []
        gate_scores = _read_json(rd / "gate" / "scores.json") or []
        meta = _read_json(rd / "meta_prompt.json") or result.get("meta_prompt") or {}
        fail_patch_raw = _read_json(rd / "patch" / "minibatch_fail_000.json")
        succ_patch_raw = _read_json(rd / "patch" / "minibatch_succ_000.json")
        merged = _read_json(rd / "patch" / "merged_patch.json") or result.get("merged_patch")
        fail_inner = (fail_patch_raw or {}).get("patch") or fail_patch_raw
        succ_inner = (succ_patch_raw or {}).get("patch") or succ_patch_raw
        meta_in = meta.get("previous") or ""
        meta_out = meta.get("meta_prompt_content") or ""

        star = " ★" if best_step == ridx else ""
        traj = _traj_from_scores(rollout_scores)
        meta_sec = format_meta_prompt_context(meta_in).strip()
        n_hard1 = sum(1 for s in rollout_scores if s.get("hard"))
        split_note = "相对 batch 均值" if use_rel else f"final≥{hard_th}"
        budget_text = (
            f"最多产出 **{edit_budget}** 条 edits。\n\n"
            f"- 分流规则：{split_note}\n"
            f"- 本轮 hard=1（success）样本数：**{n_hard1}**\n"
            f"- success analyst：{'有' if succ_inner else '**未触发**'}"
        )

        reflect_instr = _read(_PROJECT_ROOT / "promptopt" / "prompts" / "reflect_analyst_error.md")
        reflect_instr_view = re.sub(r"\{[a-z_]+\}", "‹注入›", reflect_instr)

        step1 = _step(
            "1",
            "当前提示词（本轮 state）",
            f'<div class="sec-body" style="max-height:220px">{_md(input_prompt)}</div>',
        )
        rollout_imgs = "".join(
            _img(root, f"rounds/{rd.name}/rollout/{s.get('id')}.png", f"{s.get('id')} {_fmt(s.get('final_score'))}")
            for s in rollout_scores
        )
        step2 = _step(
            "2",
            f"Rollout 生图 + 审美 · soft={_fmt(result.get('rollout_soft'))}",
            f'<div class="thumbs">{rollout_imgs or "<span class=\'muted\'>无</span>"}</div>'
            f'<p class="muted" style="margin:.4rem 0 0">轨迹含正/负向反馈分栏；分流={split_note}</p>',
        )
        reflect_body = "".join([
            _section("other", "Reflect 模板规则", reflect_instr_view, "工程指令骨架"),
            _section("scope", "注入 · prompt_scope", scope),
            _section("design", "注入 · 约束锚点（设计要求）", design_section),
            _section("meta", "注入 · 上轮 Meta 记忆", meta_sec or "（首轮为空）"),
            _section("budget", "注入 · 编辑预算 / 分流", budget_text),
            _section("prompt", "注入 · current_prompt", input_prompt),
            _section("traj", "注入 · trajectories（正/负反馈）", traj),
            f'<div class="block-label" style="margin-top:.5rem">Reflect 输出 → patch</div>{_edits_summary(fail_inner)}',
        ])
        step3 = _step("3", "Reflect LLM", reflect_body, eng=True)

        if fail_inner:
            patches_json = json.dumps(
                [x for x in (fail_inner, succ_inner) if x],
                ensure_ascii=False,
                indent=2,
            )
            merge_body = "".join([
                _section("scope", "注入 · prompt_scope", scope),
                _section("design", "注入 · 约束锚点", design_section),
                _section("meta", "注入 · Meta 记忆", meta_sec or "（空）"),
                _section("prompt", "注入 · current_prompt", input_prompt),
                _section("patch", "注入 · analyst patches", patches_json, as_md=False),
                _section("budget", "注入 · edit_budget", str(edit_budget)),
                f'<div class="block-label" style="margin-top:.5rem">Merge 输出</div>{_edits_summary(merged or fail_inner)}',
            ])
        else:
            merge_body = '<p class="muted">本轮无 Merge</p>'
        step4 = _step("4", "Merge LLM", merge_body, eng=True)

        step5 = _step(
            "5",
            "Apply patch → Prompt Diff（本轮改动标记）",
            f'<p class="muted" style="margin:0 0 .4rem">红=删/旧 · 绿=增/新</p>{_diff_html(input_prompt, candidate)}'
            + (
                f'<div class="block-label" style="margin-top:.55rem">candidate_prompt</div>'
                f'<div class="sec-body" style="max-height:200px">{_md(candidate)}</div>'
                if candidate else ""
            ),
        )

        gate_imgs = "".join(
            _img(root, f"rounds/{rd.name}/gate/{s.get('id')}.png", f"{s.get('id')} {_fmt(s.get('final_score'))}")
            for s in gate_scores
        )
        step6 = _step(
            "6",
            f"Gate 验证 · {_fmt(result.get('gate_soft'))} → {action}",
            f'<div class="thumbs">{gate_imgs or "<span class=\'muted\'>无 / skip</span>"}</div>'
            f'<p class="muted" style="margin:.4rem 0 0">规则：gate_soft &gt; current_soft 则 accept</p>',
        )
        meta_body = "".join([
            _section("design", "注入 · 约束锚点", design_section),
            _section("prompt", "注入 · 初始 prompt 摘要", initial_prompt[:800]),
            _section("meta", "注入 · previous", meta_in or "（空）"),
            _section(
                "other",
                "注入 · 本轮 digest",
                f"## Round {ridx}\n- action: **{action}**\n- rollout: {_fmt(result.get('rollout_soft'))}\n"
                f"- gate: {_fmt(result.get('gate_soft'))}\n- best: {_fmt(result.get('best_score'))}",
            ),
            _section("meta", "Meta 输出", meta_out or "（空）", meta.get("reasoning") or ""),
        ])
        step7 = _step("7", "Meta LLM", meta_body, eng=True)

        cols.append(f"""
<div class="col">
  <div class="col-head">
    <h2>R{ridx}{star} <span class="badge {badge}">{_esc(action)}</span></h2>
    <div class="scores">roll {_esc(_fmt(result.get('rollout_soft')))} → gate {_esc(_fmt(result.get('gate_soft')))} · best {_esc(_fmt(result.get('best_score')))}</div>
  </div>
  <div class="col-body">
    <p class="note">顺序：业务 prompt → Rollout → Reflect → Merge → <b>Diff</b> → Gate → Meta</p>
    {step1}{step2}{step3}{step4}{step5}{step6}{step7}
  </div>
</div>
""")

    cols.append(f"""
<div class="col init">
  <div class="col-head">
    <h2>Best</h2>
    <div class="scores">{_esc(_fmt((summary.get('best') or {}).get('final_score')))} @R{_esc(best_step)}</div>
  </div>
  <div class="col-body">
    <div class="thumbs">
      {_img(root, 'initial/image.png', 'initial')}
      {_img(root, 'best/image.png', 'best')}
    </div>
    <div class="block"><div class="block-label">Best vs Initial Diff</div>
      {_diff_html(initial_prompt, _read(root / 'best' / 'prompt.md'))}
    </div>
    <div class="block"><div class="block-label">Best prompt</div>
      <div class="sec-body" style="max-height:360px">{_md(_read(root / 'best' / 'prompt.md'))}</div>
    </div>
  </div>
</div>
""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>流程 Trace · {_esc(root.name)}</title>
<style>{CSS}</style>
</head>
<body>
{embed_tabs}
<header class="top">
  <h1>全流程 Trace · {_esc(root.name)} · case #{_esc(case.get('index'))}</h1>
  <span class="chip">Init {_esc(_fmt((summary.get('initial') or {}).get('final_score')))}</span>
  <span class="chip up">Best {_esc(_fmt((summary.get('best') or {}).get('final_score')))} @R{_esc(best_step)}</span>
  <div class="legend">
    <span><del>删/旧</del> <ins>增/新</ins></span>
    <span style="color:var(--tag-design)">■ design</span>
    <span style="color:var(--tag-traj)">■ trajectories</span>
    <span style="color:var(--tag-prompt)">■ prompt</span>
  </div>
  <span class="meta">← 横向滚动 · 每列一轮 · Diff 标记每轮改动</span>
</header>
<div class="board">
{''.join(cols)}
</div>
{LIGHTBOX_HTML}
{LIGHTBOX_JS}
</body>
</html>
"""


def generate_run_report(out_root: str | Path) -> str:
    """写入 out_root/report.html（新格式，替代旧竖版 report）。"""
    root = Path(out_root).resolve()
    page = build_run_report_html(root)
    out = root / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"  [Report] {out}")
    return str(out)


def build_session_dashboard_html(manifest: dict[str, Any], *, outputs_root: Path | None = None) -> str:
    """多任务 Tab：点 Tab 切换 iframe 加载各 run 的 report.html。"""
    outputs_root = outputs_root or (_PROJECT_ROOT / "outputs")
    runs = [r for r in (manifest.get("runs") or []) if r.get("out_root")]
    tabs: list[str] = []
    first_href = ""
    for i, row in enumerate(runs):
        out = Path(row["out_root"])
        try:
            href = str((out / "report.html").relative_to(outputs_root)).replace("\\", "/")
        except ValueError:
            href = (out / "report.html").as_posix()
        if i == 0:
            first_href = href
        label = row.get("label") or f"#{row.get('index')}"
        case_i = row.get("case_index", "—")
        best = row.get("best_score", "—")
        st = row.get("status", "?")
        active = " active" if i == 0 else ""
        tabs.append(
            f'<button type="button" class="task-tab{active}" data-src="{_esc(href)}">'
            f'{_esc(label)} · case {_esc(case_i)}'
            f'<span class="sc">{_esc(best)}</span>'
            f' · {_esc(st)}</button>'
        )

    completed = manifest.get("completed", 0)
    total = manifest.get("total", 0)
    refresh = '<meta http-equiv="refresh" content="60"/>' if completed < total else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
{refresh}
<title>训练报告 · {_esc(manifest.get('session_id', ''))}</title>
<style>{CSS}
body{{overflow:hidden}}
</style>
</head>
<body>
<header class="top">
  <h1>训练报告 · {_esc(manifest.get('session_id', ''))}</h1>
  <span class="chip">{_esc(str(completed))}/{_esc(str(total))} 完成</span>
  <span class="meta">更新 {_esc(manifest.get('updated_at', ''))} · 顶部 Tab 切换任务</span>
</header>
<nav class="task-tabs" id="tabs">{''.join(tabs) or '<span class="muted">暂无任务</span>'}</nav>
<div class="frame-wrap">
  <iframe id="frame" src="{_esc(first_href)}" title="run report"></iframe>
</div>
<script>
(function(){{
  var tabs = document.querySelectorAll('.task-tab');
  var frame = document.getElementById('frame');
  tabs.forEach(function(btn){{
    btn.addEventListener('click', function(){{
      tabs.forEach(function(b){{ b.classList.remove('active'); }});
      btn.classList.add('active');
      frame.src = btn.getAttribute('data-src');
    }});
  }});
}})();
</script>
</body>
</html>
"""


def write_session_dashboard(session_dir: str | Path, manifest: dict[str, Any]) -> str:
    session_dir = Path(session_dir)
    body = build_session_dashboard_html(manifest)
    dash = session_dir / "dashboard.html"
    dash.write_text(body, encoding="utf-8")
    latest = _PROJECT_ROOT / "outputs" / "LATEST_DASHBOARD.html"
    latest.write_text(body, encoding="utf-8")
    return str(dash)
