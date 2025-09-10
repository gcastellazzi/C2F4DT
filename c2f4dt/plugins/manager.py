from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PluginMeta:
    """Metadata for a plugin."""
    name: str
    order: int = 100
    requires: tuple[str, ...] = ()

class PluginManager:
    """Discover and manage plugins from an `extensions` directory."""

    def __init__(self, parent, extensions_dir: Optional[str] = None) -> None:
        self.parent = parent
        self.extensions_dir = extensions_dir or os.path.join(os.getcwd(), "extensions")
        self._plugins: list[PluginMeta] = []
        self._discover()

    def _discover(self) -> None:
        if not os.path.isdir(self.extensions_dir):
            return
        for entry in os.listdir(self.extensions_dir):
            path = os.path.join(self.extensions_dir, entry)
            if os.path.isdir(path):
                meta = self._load_meta(path)
                if meta:
                    self._plugins.append(meta)
        # Sort by order
        self._plugins.sort(key=lambda m: m.order)

    def _load_meta(self, plugin_dir: str) -> Optional[PluginMeta]:
        meta_py = os.path.join(plugin_dir, "plugin.py")
        if not os.path.isfile(meta_py):
            return None
        spec = importlib.util.spec_from_file_location("plugin_meta", meta_py)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[misc]
            # Expecting PLUGIN = PluginMeta(...)
            if hasattr(mod, "PLUGIN"):
                plugin = getattr(mod, "PLUGIN")
                if isinstance(plugin, PluginMeta) or (
                    hasattr(plugin, "name") and hasattr(plugin, "order") and hasattr(plugin, "requires")
                ):
                    # Coerce to PluginMeta if it's a simple namespace
                    return PluginMeta(name=str(plugin.name), order=int(plugin.order), requires=tuple(plugin.requires))
        return None

    def available_plugins(self) -> List[str]:
        return [p.name for p in self._plugins]
