"""Reversible documentation archival; no experiment code or data are changed.

Run with --apply once, then --verify. Uses only the Python standard library.
The original root progress stays in place because the latest external closeout
binds it. Other root Markdown files become byte-preserving compatibility links.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from datetime import datetime, timezone
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive/documentation_20260915"
MANIFEST = OUT / "manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    part = path.with_suffix(path.suffix + ".partial")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    part.replace(path)


def category(path):
    name = path.name
    if path.parent != ROOT:
        return str(path.parent.relative_to(ROOT))
    if name.startswith("阶段6"):
        return "mainline/stage6_history"
    if name.startswith("阶段") or name.startswith("迁移"):
        return "mainline/stage_history"
    if name.startswith(("04_", "05_对抗", "06_")):
        return "theory_and_proposals"
    return "project_history"


def apply():
    if MANIFEST.exists():
        raise SystemExit("Already applied; use --verify.")
    scopes = [ROOT, *sorted((ROOT / "revisions").glob("*/Codex")),
              *sorted((ROOT / "external_benchmarks").glob("*"))]
    files = sorted({p for scope in scopes for p in scope.glob("*.md")
                    if p.name != "README.md" or scope != ROOT})
    OUT.mkdir(parents=True, exist_ok=True)
    entries = [{"original": str(p.relative_to(ROOT)), "sha256": sha(p),
                "size": p.stat().st_size, "mode": p.stat().st_mode & 0o777,
                "original_resolved": str(p.resolve()), "category": category(p),
                "action": "archive_and_link" if p.parent == ROOT and
                p.name != "01_WDRO-REST_项目进度.md" else "preserve_in_place"}
               for p in files]
    archive = OUT / "original_documents.tar.gz"
    temporary = archive.with_suffix(archive.suffix + ".partial")
    with tarfile.open(temporary, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    with tarfile.open(temporary, "r:gz") as tar:
        for item in entries:
            assert hashlib.sha256(tar.extractfile(item["original"]).read()).hexdigest() == item["sha256"]
    temporary.replace(archive)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "root": str(ROOT), "archive_sha256": sha(archive),
                "root_markdown_before": len(list(ROOT.glob("*.md"))),
                "source_document_count": len(entries),
                "source_document_bytes": sum(x["size"] for x in entries),
                "archive_bytes": archive.stat().st_size,
                "entries": entries,
                "policy": "Source bytes preserved; root historical docs moved with compatibility symlinks. Final revision reports and bound progress remain in place."}
    write_json(MANIFEST, manifest)
    for item in entries:
        if item["action"] != "archive_and_link":
            continue
        source = ROOT / item["original"]
        dest = OUT / "originals" / item["category"] / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        source.replace(dest)
        source.symlink_to(os.path.relpath(dest, source.parent))
        item["archived_path"] = str(dest.relative_to(ROOT))
    write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("source_document_count", "source_document_bytes", "archive_bytes")}, ensure_ascii=False))


def verify():
    manifest = json.loads(MANIFEST.read_text())
    problems = []
    archive = OUT / "original_documents.tar.gz"
    if sha(archive) != manifest["archive_sha256"]:
        problems.append("archive hash")
    source_ok = 0
    tar_ok = 0
    with tarfile.open(archive, "r:gz") as tar:
        for item in manifest["entries"]:
            p = ROOT / item["original"]
            if p.is_file() and sha(p) == item["sha256"]:
                source_ok += 1
            else:
                problems.append("source hash: " + item["original"])
            if hashlib.sha256(tar.extractfile(item["original"]).read()).hexdigest() == item["sha256"]:
                tar_ok += 1
            else:
                problems.append("tar hash: " + item["original"])
            if item["action"] == "preserve_in_place" and str(p.resolve()) != item["original_resolved"]:
                problems.append("preserved resolved path: " + item["original"])
    docs = [ROOT / "README.md", ROOT / "02_新对话执行提示词.md", OUT / "README.md", *sorted((ROOT / "docs").rglob("*.md")),
            *[ROOT / d / "README.md" for d in ("mainline", "improvements", "external", "archive")]]
    checked_links = []
    # Only Markdown links in the new authored navigation are in scope. Old
    # historical prose contains absent documents and dated runtime paths.
    for p in docs:
        if not p.is_file():
            problems.append("missing new document: " + str(p))
            continue
        for raw in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-zA-Z]+://", target):
                continue
            target = re.sub(r":\d+$", "", unquote(target))
            resolved = Path(target) if target.startswith("/") else p.parent / target
            ok = resolved.exists()
            checked_links.append({"document": str(p.relative_to(ROOT)), "target": raw, "exists": ok})
            if not ok:
                problems.append("broken link: " + str(p.relative_to(ROOT)) + " -> " + raw)
    navigation_links = []
    for folder in ("mainline", "improvements", "external", "archive"):
        for p in (ROOT / folder).iterdir():
            if p.is_symlink():
                navigation_links.append({"path": str(p.relative_to(ROOT)), "target": os.readlink(p), "exists": p.exists()})
                if not p.exists():
                    problems.append("broken navigation symlink: " + str(p))
    bound_file = OUT / "existing_final_document_bindings.json"
    final_bound_documents = []
    if bound_file.exists():
        for item in json.loads(bound_file.read_text()):
            p = Path(item["path"])
            item["actual"] = sha(p) if p.is_file() else None
            item["matches"] = item["actual"] == item["expected"]
            final_bound_documents.append(item)
            if not item["matches"]:
                problems.append("existing final document binding: " + str(p))
        write_json(bound_file, final_bound_documents)
    historical_file = OUT / "historical_link_review.json"
    historical_count = 0
    if historical_file.exists():
        historical = json.loads(historical_file.read_text())
        for item in historical["links"]:
            item["exists"] = Path(item["resolved_from_original_path"]).exists()
        historical["existing"] = sum(x["exists"] for x in historical["links"])
        historical["missing"] = [x for x in historical["links"] if not x["exists"]]
        historical_count = len(historical["links"])
        if historical["missing"]:
            problems.append("historical links missing: " + str(len(historical["missing"])))
        write_json(historical_file, historical)
    result = {"status": "pass" if not problems else "fail",
              "verified_at": datetime.now(timezone.utc).isoformat(),
              "original_documents_hash_verified": source_ok,
              "archive_documents_hash_verified": tar_ok,
              "root_regular_markdown_after": sum(not p.is_symlink() for p in ROOT.glob("*.md")),
              "root_compatibility_markdown_links": sum(p.is_symlink() for p in ROOT.glob("*.md")),
              "new_authored_document_count": len(docs),
              "new_document_markdown_links_checked": len(checked_links),
              "historical_markdown_links_checked": historical_count,
              "existing_final_document_bindings_checked": len(final_bound_documents),
              "links": checked_links, "navigation_symlinks": navigation_links,
              "problems": problems,
              "scope": "Document bytes, archive recoverability and new navigation only; not a replay of scientific numerical audits."}
    write_json(OUT / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("links", "navigation_symlinks")}, ensure_ascii=False, indent=2))
    return not problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply()
    if args.verify:
        raise SystemExit(0 if verify() else 1)
