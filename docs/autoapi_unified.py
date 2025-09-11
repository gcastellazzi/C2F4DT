#!/usr/bin/env python3
"""Auto-generate MkDocs API pages for Python packages via mkdocstrings.

This script discovers Python modules in your package (e.g. `cloud2fem`),
and generates Markdown pages that reference each module with mkdocstrings.

Two output modes are supported:
  1) pages  — one Markdown file per module under `docs/api/`
  2) single — a single `docs/api_reference.md` listing all the modules

It can optionally update your `mkdocs.yml` navigation to include the generated pages.

Usage:
    python autoapi_unified.py --package cloud2fem --mode pages --docs-dir docs --update-nav

Requires in your MkDocs setup:
  - mkdocs-material (recommended)
  - mkdocstrings
  - mkdocstrings-python

Author: Unified by ChatGPT for Giovanni (C2F4DT)
"""

from __future__ import annotations
import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Dict, Tuple
import argparse
import sys

# ---------------------------
# Configuration dataclass
# ---------------------------

@dataclass
class Config:
    """Configuration for API generation.

    Attributes:
        package (str): Top-level package to scan (e.g., 'cloud2fem').
        docs_dir (Path): Root docs folder (usually 'docs').
        mode (str): 'pages' for one file per module; 'single' for one file with all modules.
        out_dir (Path): Output directory for pages (default: docs/api). Ignored in 'single' mode.
        api_index (Path): Single file path for 'single' mode (default: docs/api_reference.md).
        include (List[str]): Glob patterns to include modules.
        exclude (List[str]): Glob patterns to exclude modules.
        update_nav (bool): If True, update mkdocs.yml navigation automatically.
        mkdocs_yml (Path): Path to mkdocs.yml (attempt auto-detect if not provided).
        title (str): Navigation title for API section.
        heading_level (int): Heading level for per-module titles (if needed).
    """
    package: str
    docs_dir: Path = Path("docs")
    mode: str = "pages"  # or "single"
    out_dir: Path = Path("docs/api")
    api_index: Path = Path("docs/api_reference.md")
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=lambda: ["*.tests*", "*.conftest*", "*.tmp*", "*.examples*"])
    update_nav: bool = False
    mkdocs_yml: Optional[Path] = None
    title: str = "API Reference"
    heading_level: int = 1


# ---------------------------
# Utilities
# ---------------------------

def discover_modules(pkg: str, root: Path) -> List[str]:
    """Discover importable modules under `pkg` starting at project root.

    Args:
        pkg: Top-level package name (e.g., 'cloud2fem').
        root: Project root path where the package directory lives.

    Returns:
        List[str]: Fully-qualified module names (e.g., 'cloud2fem.ops.slices').
    """
    pkg_dir = root / pkg
    if not pkg_dir.exists():
        raise FileNotFoundError(f"Package directory not found: {pkg_dir}")
    mods: List[str] = []
    for path in pkg_dir.rglob("*.py"):
        rel = path.relative_to(root)
        # skip __pycache__ or private build files
        if any(part == "__pycache__" for part in rel.parts):
            continue
        if path.name == "__init__.py":
            # represent as package module (folder itself)
            dotted = ".".join((rel.parent).with_suffix("").parts)
        else:
            dotted = ".".join(rel.with_suffix("").parts)
        # Make sure it starts with the package name
        if not dotted.startswith(pkg + ".") and dotted != pkg:
            continue
        # Normalize
        dotted = dotted.replace(os.sep, ".").strip(".")
        if dotted and dotted not in mods:
            mods.append(dotted)
    mods.sort()
    return mods


def apply_filters(mods: Iterable[str], include: List[str], exclude: List[str]) -> List[str]:
    """Filter discovered modules by include/exclude globs.

    Args:
        mods: Iterable of module names.
        include: Glob patterns to include; if empty, include all.
        exclude: Glob patterns to exclude.

    Returns:
        List[str]: Filtered module names.
    """
    selected = list(mods)
    if include:
        selected = [m for m in selected if any(fnmatch.fnmatch(m, pat) for pat in include)]
    if exclude:
        selected = [m for m in selected if not any(fnmatch.fnmatch(m, pat) for pat in exclude)]
    return sorted(selected)


def ensure_dir(p: Path) -> None:
    """Create a directory if it does not exist (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


def write_module_page(md_path: Path, module: str) -> None:
    """Write a mkdocstrings stub page for a module.

    The page uses the ::: directive, which mkdocstrings resolves at build time.

    Args:
        md_path: Output .md file path.
        module: Fully-qualified module name.
    """
    content = f"""---
title: {module}
---

::: {module}
"""
    md_path.write_text(content, encoding="utf-8")


def write_single_index(index_path: Path, modules: List[str], title: str = "API Reference") -> None:
    """Write a single index page listing all modules with mkdocstrings blocks.

    Args:
        index_path: Markdown file output path.
        modules: List of module names to include.
        title: Page title.
    """
    lines = [f"# {title}", ""]
    for m in modules:
        lines.append(f"## {m}")  # simple h2 for readability
        lines.append("""
::: {m}
""")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def update_mkdocs_nav(cfg: Config, modules: List[str]) -> None:
    """Update mkdocs.yml navigation for API pages.

    - For 'pages' mode: create a nested section with one entry per module.
    - For 'single' mode: point to the single `api_reference.md` file.

    Args:
        cfg: Configuration object.
        modules: List of module names used to build nav entries.
    """
    mk_path = cfg.mkdocs_yml or Path("mkdocs.yml")
    if not mk_path.exists():
        logging.warning("mkdocs.yml not found; skipping nav update.")
        return
    data = yaml.safe_load(mk_path.read_text(encoding="utf-8")) or {}
    nav = data.get("nav", [])

    # Remove existing API Reference entries (naively by title match)
    new_nav = []
    for entry in nav:
        if isinstance(entry, dict) and cfg.title in entry:
            # skip old API section
            continue
        new_nav.append(entry)

    if cfg.mode == "single":
        api_entry = {cfg.title: str(cfg.api_index.relative_to(cfg.docs_dir))}
    else:
        # Build nested entries for modules
        api_items = []
        for m in modules:
            md = cfg.out_dir / f"{m}.md"
            rel = str(md.relative_to(cfg.docs_dir))
            api_items.append({m: rel})
        api_entry = {cfg.title: api_items}

    new_nav.append(api_entry)
    data["nav"] = new_nav
    mk_path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logging.info("Updated navigation in %s", mk_path)


def run(cfg: Config) -> None:
    """Main entry point to generate API docs.

    Steps:
      1) Discover modules in package.
      2) Filter by include/exclude.
      3) Write pages (per-module) or single index.
      4) Optionally update mkdocs navigation.

    Raises:
      FileNotFoundError: If the package directory is missing.
    """
    logging.info("Scanning package '%s'…", cfg.package)
    project_root = Path.cwd()
    modules = discover_modules(cfg.package, project_root)
    modules = apply_filters(modules, cfg.include, cfg.exclude)

    if cfg.mode == "single":
        ensure_dir(cfg.docs_dir)
        write_single_index(cfg.api_index, modules, title=cfg.title)
        logging.info("Wrote %s", cfg.api_index)
    else:
        ensure_dir(cfg.out_dir)
        for m in modules:
            md_path = cfg.out_dir / f"{m}.md"
            ensure_dir(md_path.parent)
            write_module_page(md_path, m)
        logging.info("Wrote %d module pages under %s", len(modules), cfg.out_dir)

    if cfg.update_nav:
        update_mkdocs_nav(cfg, modules)


def parse_args(argv: Optional[List[str]] = None) -> Config:
    """Parse CLI arguments into a Config object.

    Args:
        argv: Optional argument list. If None, defaults to sys.argv[1:].

    Returns:
        Config: Parsed configuration.
    """
    parser = argparse.ArgumentParser(
        prog="autoapi_unified",
        description="Generate MkDocs API pages (mkdocstrings) for a Python package.",
    )
    parser.add_argument("--package", required=True, help="Top-level package name (e.g., cloud2fem)")
    parser.add_argument("--docs-dir", default="docs", help="MkDocs docs directory (default: docs)")
    parser.add_argument("--mode", choices=["pages", "single"], default="pages", help="Output mode: pages|single")
    parser.add_argument("--out-dir", default="docs/api", help="Output directory for 'pages' mode (default: docs/api)")
    parser.add_argument("--api-index", default="docs/api_reference.md", help="Single index path for 'single' mode")
    parser.add_argument("--include", action="append", default=[], help="Glob pattern(s) to include modules (repeatable)")
    parser.add_argument("--exclude", action="append", default=None, help="Glob pattern(s) to exclude modules (repeatable)")
    parser.add_argument("--update-nav", action="store_true", help="Update mkdocs.yml navigation")
    parser.add_argument("--mkdocs-yml", default=None, help="Path to mkdocs.yml (auto-detect if omitted)")
    parser.add_argument("--title", default="API Reference", help="Navigation title for API section")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging" )

    args = parser.parse_args(argv or sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    exclude = args.exclude if args.exclude is not None else ["*.tests*", "*.conftest*", "*.tmp*", "*.examples*"]

    cfg = Config(
        package=args.package,
        docs_dir=Path(args.docs_dir),
        mode=args.mode,
        out_dir=Path(args.out_dir),
        api_index=Path(args.api_index),
        include=args.include,
        exclude=exclude,
        update_nav=args.update_nav,
        mkdocs_yml=Path(args.mkdocs_yml) if args.mkdocs_yml else None,
        title=args.title,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
