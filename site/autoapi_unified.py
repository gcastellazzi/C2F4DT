#!/usr/bin/env python3
"""Auto-generate MkDocs API pages for Python packages via mkdocstrings.

This script discovers Python modules in your package (e.g. `cloud2fem`),
and generates Markdown pages that reference each module with mkdocstrings.

Two output modes are supported:
  1) pages  — one Markdown file per module under `docs/api/`
  2) single — a single `docs/api_reference.md` listing all the modules

It can optionally update your `mkdocs.yml` navigation to include the generated pages.

Usage:
    python docs/autoapi_unified.py --package c2f4dt --mode pages --docs-dir docs --update-nav

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
import fnmatch
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Dict, Tuple
import argparse
import sys

# Write nested Markdown paths (dots -> folders). Set to False to write flat files.
NESTED_PATHS = True

def module_to_md_path(module: str, out_dir: Path, nested: bool = NESTED_PATHS) -> Path:
    """Map dotted module path to Markdown file path under out_dir.
    
    Args:
        module: Dotted module, e.g. 'c2f4dt.ui.viewer3d'.
        out_dir: Base output directory, e.g. docs/api.
        nested: If True, create folders from dots.
    """
    return ((out_dir / Path(*module.split("."))) if nested else (out_dir / f"{module}" )).with_suffix(".md")

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
        lines.append(f"## {m}")
        lines.append(f"::: {m}")  # <-- use f-string so mkdocstrings resolves it
    index_path.write_text("\n".join(lines), encoding="utf-8")


def write_api_toc_index(out_dir: Path, modules: List[str], title: str = "API Index") -> None:
    """Write a simple API index page with links to all generated module pages.

    The file is created at `<out_dir>/index.md` and links use paths relative to `out_dir`.

    Args:
        out_dir: The `docs/api` directory.
        modules: List of module names.
        title: Page title to display.
    """
    lines = [f"# {title}", "", "> Auto-generated index of API modules.", ""]
    for m in modules:
        md_path = module_to_md_path(m, out_dir)
        rel = md_path.relative_to(out_dir)
        lines.append(f"- [{m}]({rel.as_posix()})")
    ensure_dir(out_dir)
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_mkdocstrings_config(data: Dict) -> None:
    """Migrate older mkdocstrings config keys and ensure Python handler uses options/paths.
    
    - Moves handlers.python.rendering -> handlers.python.options.
    - Ensures handlers.python.options exists.
    - Ensures handlers.python.paths includes 'src' if a src/ layout is detected.
    Modifies `data` in place.
    """
    plugins = data.get("plugins")
    if not plugins:
        return
    # plugins can be a list or a dict (Material flattens names). Handle both.
    def iter_items(plg):
        if isinstance(plg, dict):
            return [(k, v) for k, v in plg.items()]
        if isinstance(plg, list):
            # entries are either strings or {name: config}
            pairs = []
            for item in plg:
                if isinstance(item, str):
                    pairs.append((item, None))
                elif isinstance(item, dict):
                    [(k, v)] = item.items()
                    pairs.append((k, v))
            return pairs
        return []
    items = iter_items(plugins)
    for name, cfg in items:
        if name == "mkdocstrings" and isinstance(cfg, dict):
            handlers = cfg.get("handlers", {})
            py = handlers.get("python") if isinstance(handlers, dict) else None
            if isinstance(py, dict):
                # migrate rendering -> options
                rendering = py.pop("rendering", None)
                options = py.get("options")
                if rendering:
                    if not isinstance(options, dict):
                        options = {}
                    options.update(rendering)
                    py["options"] = options
                # ensure paths
                paths = py.get("paths")
                if not paths:
                    py["paths"] = ["src"] if Path("src").exists() else ["."]
            # write back (dict case is by ref; list case we need to update the original item)
            if isinstance(plugins, list):
                for i, item in enumerate(plugins):
                    if isinstance(item, dict) and "mkdocstrings" in item:
                        plugins[i] = {"mkdocstrings": cfg}
                        break
            elif isinstance(plugins, dict):
                plugins["mkdocstrings"] = cfg


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

    # Normalize mkdocstrings config so it doesn't crash on newer versions
    normalize_mkdocstrings_config(data)

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
        # Build nested entries for modules (include an index page first)
        api_items = []
        api_index_rel = str((cfg.out_dir / "index.md").relative_to(cfg.docs_dir))
        api_items.append({"Index": api_index_rel})
        for m in modules:
            md = module_to_md_path(m, cfg.out_dir)
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
    root = project_root
    # Auto-detect src/ layout
    if (project_root / "src" / cfg.package).exists():
        root = project_root / "src"

    modules = discover_modules(cfg.package, root)
    modules = apply_filters(modules, cfg.include, cfg.exclude)

    # Exclude common non-importable assets to avoid mkdocstrings errors
    modules = [m for m in modules if not m.startswith(f"{cfg.package}.assets.")]

    if cfg.mode == "single":
        ensure_dir(cfg.docs_dir)
        write_single_index(cfg.api_index, modules, title=cfg.title)
        logging.info("Wrote %s", cfg.api_index)
    else:
        ensure_dir(cfg.out_dir)
        for m in modules:
            md_path = module_to_md_path(m, cfg.out_dir)
            ensure_dir(md_path.parent)
            write_module_page(md_path, m)
        # Also write an API index with links to all modules
        write_api_toc_index(cfg.out_dir, modules, title="API Index")
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
