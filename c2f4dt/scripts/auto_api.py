#!/usr/bin/env python3
"""
Auto-generate MkDocs API pages and nav for the project.

This script scans one or more Python packages (e.g., `c2f4dt` and `extensions`),
creates a Markdown file per module with a mkdocstrings directive, and updates
`mkdocs.yml` to include a nested API navigation up to a configurable depth.

Usage
-----
    python scripts/auto_api.py \
        --package c2f4dt --package extensions \
        --out docs/api \
        --mkdocs mkdocs.yml \
        --max-depth 5 \
        --title "API Reference"

Notes
-----
- Requires `PyYAML` (pip install pyyaml).
- Renders with `mkdocstrings[python]`; ensure it's in requirements.txt.
- The script creates/overwrites Markdown under `--out`.
- The script updates mkdocs.yml: it appends/overwrites an `API` entry under `nav`;
  other sections are preserved.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print("PyYAML is required. Please: pip install pyyaml", file=sys.stderr)
    raise


@dataclass
class ModuleDoc:
    """Represents a module to document and its target markdown path."""
    module: str
    md_path: Path


# ---------------------- Discovery ---------------------- #
def discover_modules(package: str) -> List[str]:
    """Return a flat list of importable module names under `package`.

    Args:
        package: Top-level package name (must be importable).

    Returns:
        Sorted list of fully-qualified module names.

    Raises:
        ImportError: if the package is not importable (silenced into a warning).
    """
    mods: List[str] = []
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:  # pragma: no cover
        print(f"[auto_api] Skip '{package}': {exc}", file=sys.stderr)
        return mods

    pkg_path = Path(getattr(pkg, "__path__", [Path(pkg.__file__).parent])[0])
    prefix = pkg.__name__ + "."

    for i in pkgutil.walk_packages([str(pkg_path)], prefix=prefix):
        name = i.name
        base = name.rsplit(".", 1)[-1]
        if base.startswith("_"):  # skip private/dunder modules
            continue
        mods.append(name)

    if package not in mods:
        mods.insert(0, package)  # include the package root
    return sorted(set(mods))


# ---------------------- Rendering ---------------------- #
def md_for_module(module: str) -> str:
    """Return Markdown content with mkdocstrings directive for `module`.

    Uses mkdocstrings python handler; options tuned for Google-style docstrings.
    """
    return f"""# `{module}`

::: {module}
    handler: python
    options:
      show_source: false
      members_order: source
      docstring_style: google
      docstring_section_style: list
      show_signature: true
      show_root_heading: false
      show_category_heading: true
      separate_signature: true
      filters:
        - '!^_'
      merge_init_into_class: true
"""


def write_markdown_pages(out_dir: Path, modules: Iterable[str]) -> List[ModuleDoc]:
    """Create a Markdown file per module under `out_dir` mirroring package tree.

    Args:
        out_dir: Destination directory (e.g., `docs/api`).
        modules: Discovered module names.

    Returns:
        List of ModuleDoc with module and markdown path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[ModuleDoc] = []

    for mod in modules:
        parts = mod.split(".")
        rel_dir = Path(*parts)
        md_path = out_dir / rel_dir.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_for_module(mod), encoding="utf-8")
        results.append(ModuleDoc(module=mod, md_path=md_path))
    return results


# ---------------------- Nav building ---------------------- #
def insert_nav(mkdocs_path: Path, title: str, module_docs: List[ModuleDoc], max_depth: int) -> None:
    """Insert or update an `API` section in mkdocs.yml navigation.

    The structure mirrors the package hierarchy up to `max_depth` levels.
    """
    if mkdocs_path.exists():
        data = yaml.safe_load(mkdocs_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    # Ensure minimal config
    data.setdefault("site_name", "C2F4DT")
    data.setdefault("theme", {"name": "material"})

    # Ensure mkdocstrings plugin
    plugins = data.get("plugins", [])
    if isinstance(plugins, dict):
        plugins = [plugins]
    names = [p if isinstance(p, str) else next(iter(p)) for p in plugins]
    if "mkdocstrings" not in names:
        plugins.append("mkdocstrings")
    if "search" not in names:
        plugins.insert(0, "search")
    data["plugins"] = plugins

    # Build nested tree
    def add_to_tree(tree: Dict, mod: ModuleDoc):
        parts = mod.module.split(".")
        node = tree
        for depth, part in enumerate(parts):
            if depth >= max_depth - 1:
                break
            node = node.setdefault(part, {})
        rel = os.path.relpath(mod.md_path, mkdocs_path.parent / "docs").replace(os.sep, "/")
        node.setdefault("__pages__", []).append({parts[-1]: rel})

    tree: Dict = {}
    for md in module_docs:
        add_to_tree(tree, md)

    def to_nav_list(node: Dict) -> List:
        items = []
        for key in sorted(k for k in node.keys() if k != "__pages__"):
            items.append({key: to_nav_list(node[key])})
        for page in node.get("__pages__", []):
            items.append(page)
        return items

    api_nav = [{title: to_nav_list(tree)}]

    # Merge/replace into existing nav
    nav = data.get("nav", [])
    if not isinstance(nav, list):
        nav = []
    def is_api_entry(entry) -> bool:
        return isinstance(entry, dict) and title in entry

    nav = [e for e in nav if not is_api_entry(e)]
    nav.extend(api_nav)
    data["nav"] = nav

    mkdocs_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ---------------------- CLI ---------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API Markdown and update mkdocs.yml nav")
    parser.add_argument("--package", action="append", required=True, help="Package to document (repeatable)")
    parser.add_argument("--out", default="docs/api", help="Output directory for generated Markdown")
    parser.add_argument("--mkdocs", default="mkdocs.yml", help="Path to mkdocs.yml")
    parser.add_argument("--max-depth", type=int, default=5, help="Max nav depth (folders)")
    parser.add_argument("--title", default="API", help="Top-level nav title")
    args = parser.parse_args()

    out_dir = Path(args.out)
    mkdocs_path = Path(args.mkdocs)

    all_modules: List[str] = []
    for pkg in args.package:
        all_modules.extend(discover_modules(pkg))

    # Deduplicate preserving order
    seen = set()
    ordered: List[str] = []
    for m in all_modules:
        if m not in seen:
            ordered.append(m)
            seen.add(m)

    module_docs = write_markdown_pages(out_dir, ordered)
    insert_nav(mkdocs_path, title=args.title, module_docs=module_docs, max_depth=args.max_depth)

    print(f"[auto_api] Generated {len(module_docs)} pages under {out_dir}")
    print(f"[auto_api] Updated nav in {mkdocs_path}")


if __name__ == "__main__":
    main()