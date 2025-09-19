# -*- coding: utf-8 -*-
"""Lightweight manager for external extensions living in separate repositories.

This manager discovers extensions in *external* folders, e.g. ../C2F4DT_extensions,
without bundling them inside C2F4DT. It mirrors the plugin Manager API
(load -> get -> run), expecting each extension folder to expose:

- extension.yaml  (metadata: name, version, order, requires, entry_point, description)
- extension.py    (entry point providing `register(parent)` -> extension instance)

Search path order (first existing wins; all scanned):
1) env var C2F4DT_EXT_DIRS: colon/semicolon separated absolute paths
2) sibling ../C2F4DT_extensions relative to C2F4DT package
3) ~/.config/C2F4DT/extensions (optional convention)

Author: C2F4DT
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # optional; if missing, we still work with defaults
except Exception:
    yaml = None  # type: ignore


@dataclass
class ExtensionMeta:
    """Metadata for an extension.

    Attributes:
        name: Public name.
        version: Semantic version string.
        order: Load order (lower first).
        requires: Tuple of python import names that must import successfully.
        entry_point: Dotted entry point "module:attr" (defaults to "extension:register").
        path: Absolute folder path of the extension.
        description: Short teaser for UI tooltips.
    """
    name: str
    version: Optional[str] = None
    order: int = 100
    requires: Tuple[str, ...] = ()
    entry_point: Optional[str] = None
    path: Optional[Path] = None
    description: Optional[str] = None


class ExtensionManager:
    """Discover & run extensions that live outside of C2F4DT.

    The manager is read-only (no install). It imports modules from external
    folders by temporarily adding each extension folder to sys.path, then
    loading its `extension.py`.
    """

    def __init__(self, parent: Any, extra_dirs: Optional[List[str | Path]] = None) -> None:
        """Initialize the manager.

        Args:
            parent: Main window (host) passed to extension factories.
            extra_dirs: Optional additional search folders. Useful for tests.
        """
        self.parent = parent
        self.search_dirs = self._resolve_search_dirs(extra_dirs)
        self._metas: list[ExtensionMeta] = []
        self._instances: Dict[str, Any] = {}
        self._discover()

    # ------------------------- discovery -------------------------
    def _resolve_search_dirs(self, extra: Optional[List[str | Path]]) -> List[Path]:
        """Return list of folders to scan for extensions (existing only).

        Resolution order:
        1) Env var C2F4DT_EXT_DIRS (colon/semicolon-separated absolute or relative paths)
        2) Sibling "C2F4DT_extensions" next to the repo root (…/C2F4DT/../C2F4DT_extensions)
        3) ~/.config/C2F4DT/extensions
        4) CWD-based fallbacks (../C2F4DT_extensions and ./C2F4DT_extensions)
        5) Caller-provided extras
        """
        dirs: List[Path] = []

        def _probe(label: str, p: Path) -> None:
            try:
                r = p.expanduser().resolve()
            except Exception as exc:
                logging.debug("[extensions] %s -> %s (resolve error: %s)", label, p, exc)
                return
            exists = r.exists()
            isdir = r.is_dir()
            logging.debug("[extensions] candidate %-10s %s | exists=%s is_dir=%s", label, r, exists, isdir)
            if isdir:
                dirs.append(r)

        # 1) ENV
        env = os.environ.get("C2F4DT_EXT_DIRS") or ""
        for chunk in filter(None, [s.strip() for s in env.replace(";", ":").split(":")]):
            _probe("ENV", Path(chunk))

        # 2) Sibling of repo root: …/C2F4DT/../C2F4DT_extensions
        here = Path(__file__).resolve()
        repo_root = here.parent.parent.parent.parent  # …/C2F4DT
        # (optional) Git-root heuristic if needed
        try:
            cur = here
            for _ in range(8):  # up to 8 levels
                if (cur / ".git").exists() or (cur / "src" / "c2f4dt").exists():
                    git_root = cur
                    break
                cur = cur.parent
            else:
                git_root = None
            if git_root:
                _probe("GIT_SIB", git_root.parent.joinpath("C2F4DT_extensions"))
        except Exception:
            pass
        # Do NOT probe repo_root itself; only its sibling C2F4DT_extensions
        sibling_dev = repo_root.parent.joinpath("C2F4DT_extensions")
        _probe("SIBLING", sibling_dev)

        # 3) XDG-like
        _probe("XDG", Path.home().joinpath(".config", "C2F4DT", "extensions"))

        # 4) CWD fallbacks
        cwd = Path.cwd()
        _probe("CWD^", cwd.parent.joinpath("C2F4DT_extensions"))
        _probe("CWD", cwd.joinpath("C2F4DT_extensions"))

        # 5) extras
        if extra:
            for e in extra:
                _probe("EXTRA", Path(e))

        # De-duplicate preserving order
        uniq: List[Path] = []
        seen = set()
        for d in dirs:
            s = str(d)
            if s not in seen:
                uniq.append(d)
                seen.add(s)

        if not uniq:
            logging.warning("[extensions] No search dirs found. Set C2F4DT_EXT_DIRS or create ../C2F4DT_extensions.")
        else:
            logging.debug("[extensions] search dirs: %s", ", ".join(map(str, uniq)))
        return uniq
    
    def _discover(self) -> None:
        """Scan search dirs for extension folders and collect metadata."""
        metas: List[ExtensionMeta] = []
        for base in self.search_dirs:
            for entry in sorted(base.iterdir()):
                if not entry.is_dir():
                    continue
                meta = self._load_meta(entry)
                if meta:
                    metas.append(meta)

        # Filter by requirements
        filtered: List[ExtensionMeta] = []
        for m in metas:
            ok, missing = self._requirements_ok(m)
            if not ok:
                logging.warning("[extensions] Skipping '%s' (missing: %s)", m.name, ", ".join(missing))
                continue
            filtered.append(m)

        filtered.sort(key=lambda m: m.order)
        self._metas = filtered

    def _load_meta(self, folder: Path) -> Optional[ExtensionMeta]:
        """Load metadata from extension.yaml (or infer sane defaults)."""
        name = folder.name
        version = None
        order = 100
        requires: Tuple[str, ...] = ()
        entry_point = "extension:register"
        description = None

        yaml_path = folder / "extension.yaml"
        if yaml and yaml_path.is_file():
            try:
                data = yaml.safe_load(yaml_path.read_text()) or {}
                name = str(data.get("name") or name)
                version = str(data.get("version") or "") or None
                order = int(data.get("order", 100))
                requires = tuple(map(str, data.get("requires", ())))
                entry_point = str(data.get("entry_point") or entry_point)
                description = str(data.get("description") or "") or None
            except Exception as exc:
                logging.error("[extensions] Failed to read %s: %s", yaml_path, exc)

        # Require a python entry file to consider it an extension
        if not (folder / "extension.py").is_file():
            return None

        return ExtensionMeta(
            name=name, version=version, order=order, requires=requires,
            entry_point=entry_point, path=folder, description=description
        )

    def _requirements_ok(self, meta: ExtensionMeta) -> Tuple[bool, List[str]]:
        missing: List[str] = []
        for req in meta.requires:
            try:
                importlib.import_module(req)
            except Exception:
                missing.append(req)
        return (len(missing) == 0, missing)

    # --------------------------- loading --------------------------
    def _import_module(self, meta: ExtensionMeta) -> Optional[ModuleType]:
        """Import `extension.py` from the external folder safely."""
        if not meta.path:
            return None
        py = meta.path.joinpath("extension.py")
        if not py.is_file():
            return None
        # Ensure folder is importable for relative imports inside extension
        folder = str(meta.path)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location(f"{meta.name}.extension", py)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)  # type: ignore[misc]
            return mod
        return None

    def _resolve_entry(self, module: ModuleType, entry_point: Optional[str]):
        """Resolve `register` factory from entry_point like 'extension:register'."""
        if not entry_point:
            return getattr(module, "register", None)
        mod_part, colon, attr = entry_point.partition(":")
        if not colon:
            return getattr(module, entry_point, None)
        # 'extension:register' → attr on the same module
        target = module
        if mod_part and mod_part not in ("extension", "."):
            # allow extension to nest code as packages if desired
            target = importlib.import_module(f"{module.__name__}.{mod_part}")
        return getattr(target, attr, None)

    # ----------------------------- API ----------------------------
    def available_extensions(self) -> List[dict]:
        """Return UI-friendly items for a combo box."""
        items: List[dict] = []
        for m in self._metas:
            ok, missing = self._requirements_ok(m)
            items.append({
                "key": m.name,
                "label": f"{m.name}{(' '+m.version) if m.version else ''}",
                "enabled": ok,
                "tooltip": (m.description or m.name) + ("" if ok else f" (missing: {', '.join(missing)})"),
                "order": m.order,
            })
        return items

    def get(self, key: str):
        """Instantiate (or return cached) extension object by key."""
        if key in self._instances:
            return self._instances[key]
        meta = next((m for m in self._metas if m.name == key), None)
        if meta is None:
            return None
        module = self._import_module(meta)
        if module is None:
            return None
        factory = self._resolve_entry(module, meta.entry_point)
        if factory is None:
            logging.warning("[extensions] No entry for '%s'", meta.name)
            return None
        try:
            obj = factory(self.parent)
        except TypeError:
            obj = factory()
        self._instances[key] = obj
        return obj

    def run(self, key: str, **context) -> bool:
        """Execute an extension by calling the first available action."""
        obj = self.get(key)
        if obj is None:
            return False
        for method in ("run", "exec", "execute", "show", "__call__"):
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn(**context)
                except TypeError:
                    fn()
                return True
        return False