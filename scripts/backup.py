# 【功能描述】将日常少用的项目资产归档到 backup/archive/；运行产物可选快照到 backup/snapshots/
# 【输入】PROJECT_ROOT、MODE、ARCHIVE_PLAN、SNAPSHOT_DIRS（脚本顶部配置）
# 【输出】backup/archive/ 分类目录、backup/snapshots/<timestamp>/manifest.json

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

# ── 运行参数（直接改这里，勿用命令行 flag）────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODE = "archive"  # archive | snapshot | both
DRY_RUN = False

# 基本不用的静态/可选模块 → backup/archive/<category>/
ARCHIVE_PLAN: dict[str, list[str]] = {
    "website": [
        "index.html",
        "skillopt.html",
        "skillopt-assets",
    ],
    "docs_site": [
        "docs",
        "mkdocs.yml",
    ],
    "shell": [
        "scripts/run_alfworld.sh",
        "scripts/run_searchqa.sh",
        "scripts/run_spreadsheetbench.sh",
    ],
    "misc": [
        "requirements.txt",
        "skillopt/scheduler",
    ],
}

# 训练/实验运行产物 → backup/snapshots/<timestamp>/
SNAPSHOT_DIRS = [
    "outputs",
    "data",
    "configs/local",
]

# ── 实现 ────────────────────────────────────────────────────────────────────

ARCHIVE_ROOT = PROJECT_ROOT / "backup" / "archive"
SNAPSHOT_ROOT = PROJECT_ROOT / "backup" / "snapshots"


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _move_to_archive(src: Path, category: str) -> dict:
    dst = ARCHIVE_ROOT / category / src.name
    record = {
        "source": _rel(src),
        "target": _rel(dst),
        "category": category,
    }
    if not src.exists():
        record["status"] = "skipped"
        record["reason"] = "not found"
        return record

    if DRY_RUN:
        record["status"] = "dry_run"
        return record

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    record["status"] = "archived"
    return record


def _snapshot_dir(src: Path, snapshot_dir: Path) -> dict:
    record = {"source": _rel(src), "target": _rel(snapshot_dir / src.name)}
    if not src.exists():
        record["status"] = "skipped"
        record["reason"] = "not found"
        return record

    if DRY_RUN:
        record["status"] = "dry_run"
        return record

    dst = snapshot_dir / src.name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    record["status"] = "snapshotted"
    return record


def run_archive() -> dict:
    items: list[dict] = []
    for category, paths in ARCHIVE_PLAN.items():
        for rel_path in paths:
            items.append(_move_to_archive(PROJECT_ROOT / rel_path, category))

    manifest = {
        "kind": "archive",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "archive_root": str(ARCHIVE_ROOT),
        "items": items,
        "summary": {
            "archived": sum(1 for i in items if i.get("status") == "archived"),
            "skipped": sum(1 for i in items if i.get("status") == "skipped"),
        },
    }
    if not DRY_RUN:
        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = ARCHIVE_ROOT / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return manifest


def run_snapshot() -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = SNAPSHOT_ROOT / f"skillopt_{stamp}"
    items = [_snapshot_dir(PROJECT_ROOT / rel, snapshot_dir) for rel in SNAPSHOT_DIRS]

    manifest = {
        "kind": "snapshot",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "snapshot_dir": str(snapshot_dir),
        "items": items,
        "summary": {
            "snapshotted": sum(1 for i in items if i.get("status") == "snapshotted"),
            "skipped": sum(1 for i in items if i.get("status") == "skipped"),
        },
    }
    if not DRY_RUN:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    results: list[dict] = []
    if MODE in {"archive", "both"}:
        results.append(run_archive())
    if MODE in {"snapshot", "both"}:
        results.append(run_snapshot())

    for manifest in results:
        kind = manifest["kind"]
        summary = manifest["summary"]
        print(f"[backup:{kind}] {summary}")
        if kind == "archive":
            print(f"  → {manifest['archive_root']}")
        else:
            print(f"  → {manifest['snapshot_dir']}")


if __name__ == "__main__":
    main()
