from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional dependency; declared in project's requirements
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML missing
    yaml = None  # type: ignore


@dataclass
class PluginMeta:
    """Metadata for a plugin.

    Attributes:
        name: Public name of the plugin (also used as folder name).
        version: Optional semantic version (e.g., "0.1.0") used in UI labels.
        order: Load order priority (lower loads first). Defaults to 100.
        requires: Tuple of python import names that must be importable.
        entry_point: Dotted entry point in the form "module:attr". If not
            provided, a callable named ``load_plugin`` inside the plugin
            package/module will be used when available.
        path: Absolute path to the plugin directory on disk.
    """

    name: str
    version: Optional[str] = None
    order: int = 100
    requires: Tuple[str, ...] = ()
    entry_point: Optional[str] = None
    path: Optional[Path] = None


class PluginManager:
    """Discover and manage plugins from the ``plugins`` directory.

    The manager searches subfolders in ``c2f4dt/plugins`` (i.e. the directory
    where this file lives) unless a different directory is provided.

    It supports two metadata sources:

    1. ``plugin.yaml`` with keys {name, order, requires, entry_point}
    2. ``plugin.py`` exposing either:
       - ``PLUGIN`` compatible with :class:`PluginMeta` (name, order, requires)
       - and/or a factory ``load_plugin(parent) -> Any``

    Typical plugin layout::

        plugins/
          hello/
            __init__.py
            plugin.py          # may define PLUGIN and load_plugin()
            plugin.yaml        # optional; overrides values from plugin.py (supports name, order, requires, entry_point, version|release)

    The instantiated plugin object is returned by :meth:`load_plugins`.
    """

    def __init__(self, parent: Any, plugins_dir: Optional[Path | str] = None) -> None:
        """Initialize the manager.

        Args:
            parent: Host object (e.g., main application) passed to plugins.
            plugins_dir: Optional absolute path to the plugins root. If not
                provided, defaults to the directory where this file resides.
        """
        self.parent = parent

        # Resolve plugins directory: explicit argument wins, otherwise use smart resolver.
        if plugins_dir:
            self.plugins_dir = Path(plugins_dir).expanduser().resolve()
        else:
            self.plugins_dir = self._resolve_default_plugins_dir()

        logging.debug(
            "[PluginManager] Using plugins_dir=%s (exists=%s, is_dir=%s)",
            self.plugins_dir,
            self.plugins_dir.exists(),
            self.plugins_dir.is_dir(),
        )

        self._metas: list[PluginMeta] = []
        self._instances: Dict[str, Any] = {}
        self._discover()

    def _resolve_default_plugins_dir(self) -> Path:
        """Return the most likely plugins directory.

        Resolution order:
        1. Environment variable ``C2F4DT_PLUGINS_DIR`` if it points to an existing dir.
        2. The directory where this file lives (``.../c2f4dt/plugins``).
        3. Project root ``.../C2F4DT/plugins`` (two levels up from this file).
        4. Current working directory ``./plugins``.
        The first *existing* directory is returned; otherwise fallback to (2).
        """
        here = Path(__file__).resolve().parent

        # 1) Environment override
        env_dir = os.environ.get("C2F4DT_PLUGINS_DIR")
        if env_dir:
            env_path = Path(env_dir).expanduser().resolve()
            if env_path.is_dir():
                return env_path

        # 2) This file's folder: c2f4dt/plugins
        if here.is_dir():
            return here

        # 3) Project root /plugins (two levels up from this file)
        project_root_plugins = here.parent.parent.joinpath("plugins").resolve()
        if project_root_plugins.is_dir():
            return project_root_plugins

        # 4) CWD /plugins
        cwd_plugins = Path.cwd().joinpath("plugins").resolve()
        if cwd_plugins.is_dir():
            return cwd_plugins

        # Fallback: return where this file resides even if it does not exist (caller will log)
        return here

    # ---------------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------------
    def _discover(self) -> None:
        """Scan the plugins directory and collect metadata.

        This function populates ``self._metas`` by reading either a
        ``plugin.yaml`` or a ``plugin.py`` file for each subdirectory.
        """
        if not self.plugins_dir.is_dir():
            logging.warning(
                "Plugin directory does not exist or is not a directory: %s (exists=%s, is_dir=%s)",
                self.plugins_dir,
                self.plugins_dir.exists(),
                self.plugins_dir.is_dir(),
            )
            return

        for entry in sorted(self.plugins_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta = self._load_meta(entry)
            if meta is None:
                continue

            # Check runtime requirements (importable modules).
            if not self._requirements_ok(meta):
                logging.warning("Skipping plugin '%s' due to missing requirements: %s",
                                meta.name, ", ".join(meta.requires))
                continue

            self._metas.append(meta)

        # Sort by declared order (stable sort)
        self._metas.sort(key=lambda m: m.order)

    def _requirements_ok(self, meta: PluginMeta) -> Tuple[bool, List[str]]:
        """Check python requirements listed in plugin.yaml."""
        missing: List[str] = []
        for req in meta.requires:
            try:
                importlib.import_module(req)
            except Exception:
                missing.append(req)
        return (len(missing) == 0, missing)

    def _meta_by_name(self, name: str) -> Optional[PluginMeta]:
        for m in self._metas:
            if m.name == name:
                return m
        return None

    
    def _load_meta(self, plugin_dir: Path) -> Optional[PluginMeta]:
        """Load metadata for a single plugin directory.

        The precedence is: ``plugin.yaml`` (if PyYAML is available) then
        ``plugin.py``'s ``PLUGIN`` object.

        Args:
            plugin_dir: Absolute path to the plugin directory.

        Returns:
            A :class:`PluginMeta` or ``None`` if not a valid plugin.
        """
        meta: Optional[PluginMeta] = None

        yaml_path = plugin_dir / "plugin.yaml"
        if yaml and yaml_path.is_file():
            try:
                data = yaml.safe_load(yaml_path.read_text()) or {}
                name = str(data.get("name") or plugin_dir.name)
                order = int(data.get("order", 100))
                requires = tuple(map(str, data.get("requires", ())))
                entry_point = data.get("entry_point")
                version = str(data.get("version") or data.get("release") or "").strip() or None
                meta = PluginMeta(name=name, order=order, requires=requires,
                                  entry_point=entry_point, path=plugin_dir,
                                  version=version)
            except Exception as exc:  # robust to malformed YAML
                logging.error("Failed to read %s: %s", yaml_path, exc)

        if meta is None:
            py_path = plugin_dir / "plugin.py"
            if py_path.is_file():
                spec = importlib.util.spec_from_file_location(f"{plugin_dir.name}.plugin", py_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(mod)  # type: ignore[misc]
                    except Exception as exc:
                        logging.error("Failed to import %s: %s", py_path, exc)
                        return None

                    # Accept either an actual PluginMeta or a simple namespace
                    plugin_obj = getattr(mod, "PLUGIN", None)
                    if plugin_obj is not None:
                        name = getattr(plugin_obj, "name", plugin_dir.name)
                        order = int(getattr(plugin_obj, "order", 100))
                        requires = tuple(getattr(plugin_obj, "requires", ()))
                        version = getattr(plugin_obj, "version", None) or getattr(plugin_obj, "release", None)
                        if isinstance(version, str):
                            version = version.strip() or None
                        else:
                            version = None
                        meta = PluginMeta(name=str(name), order=order,
                                          requires=tuple(map(str, requires)), path=plugin_dir,
                                          version=version)

        if meta is not None:
            # Ensure path is set for later imports
            meta.path = plugin_dir
        return meta

    # ---------------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------------
    def _dependency_status(self, meta: PluginMeta) -> tuple[list[str], list[str]]:
        """Return (ok, missing) lists for the required imports of a plugin.

        Args:
            meta: Plugin metadata.
        """
        ok: list[str] = []
        missing: list[str] = []
        for req in meta.requires:
            try:
                importlib.import_module(req)
            except Exception:
                missing.append(req)
            else:
                ok.append(req)
        return ok, missing

    def _import_plugin_package(self, meta: PluginMeta) -> Optional[ModuleType]:
        """Import the plugin as a package/module in a stable manner.

        Strategy:
        1) If the plugin folder is a reachable package, import it as
           '<package_root>.<plugin_name>' (e.g., 'c2f4dt.plugins.units').
        2) Fallback: directly load 'plugin.py' using a spec loader,
           **registering the module in sys.modules** before executing it,
           so that decorators like @dataclass can find the module.
        """
        package_root = ".".join(__name__.split(".")[:-1])  # e.g., 'c2f4dt.plugins'
        dotted = f"{package_root}.{meta.name}"

        # 1) Attempt a "normal" import
        try:
            return importlib.import_module(dotted)
        except Exception:
            pass

        # 2) Fallback: direct import of plugin.py
        py = meta.path.joinpath("plugin.py") if meta.path else None
        if py and py.is_file():
            spec = importlib.util.spec_from_file_location(f"{meta.name}.plugin", py)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # *** CRITICAL: Register the module before executing it ***
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)  # type: ignore[misc]
                return mod
        return None

    def _resolve_entry_point(self, module: ModuleType, entry_point: Optional[str]) -> Optional[Any]:
        """
        Resolve and return the factory/class pointed to by the entry_point.

        Supported formats:
        - "attr"                 -> getattr(module, "attr")
        - ":attr"                -> getattr(module, "attr")  (explicitly same module)
        - "submod:attr"          -> import module.__name__ + ".submod" -> getattr(submod, "attr")
        - "pkg.submod:attr"      -> absolute import "pkg.submod" -> getattr(submod, "attr")
        """
        if not entry_point:
            # Fallback to a conventional default
            return getattr(module, "load_plugin", None)

        mod_part, colon, attr = entry_point.partition(":")
        if not colon:
            # Format "attr" without a colon
            return getattr(module, entry_point, None)

        # At this point, we always have "X:Y" (even if X can be empty like in ":attr")
        if not attr:
            # No attribute to resolve
            return None

        target_module = module
        if mod_part:
            # If mod_part contains dots, treat it as ABSOLUTE (e.g., "plugins.units.plugin")
            # Otherwise, treat it as RELATIVE to the package of the passed module.
            try:
                if "." in mod_part:
                    target_module = importlib.import_module(mod_part)
                else:
                    # The module is the base package; import one of its submodules
                    target_module = importlib.import_module(f"{module.__name__}.{mod_part}")
            except Exception as exc:
                logging.error(
                    "Failed to import entry submodule '%s' for '%s': %s",
                    mod_part, getattr(module, "__name__", module), exc
                )
                return None
        
        logging.debug("Entry point target_module=%s file=%s", getattr(target_module, "__name__", "?"), getattr(target_module, "__file__", "?"))
        logging.debug("Entry point attributes sample=%s", [n for n in dir(target_module) if n in ("register", "load_plugin", "UnitsPlugin")])
        obj = getattr(target_module, attr, None)
        return obj

    def load_plugins(self) -> Dict[str, Any]:
        """Instantiate and return all enabled plugins.

        Returns:
            Dict mapping plugin name to the instantiated plugin object.
        """
        instances: Dict[str, Any] = {}
        for meta in self._metas:
            module = self._import_plugin_package(meta)
            if module is None:
                logging.warning("Cannot import plugin module for '%s'", meta.name)
                continue

            factory_or_cls = self._resolve_entry_point(module, meta.entry_point)
            if factory_or_cls is None:
                logging.warning("No entry point found for plugin '%s'", meta.name)
                continue

            try:
                # Support both factory(parent) and class(parent)
                obj = factory_or_cls(self.parent)
            except TypeError:
                # Maybe it's a zero-arg factory/class
                obj = factory_or_cls()
            except Exception as exc:
                logging.error("Failed to instantiate plugin '%s': %s", meta.name, exc)
                continue

            instances[meta.name] = obj
        return instances

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------
    def available_plugins(self) -> List[str]:
        """Return the list of discovered plugin names, sorted by order."""
        return [m.name for m in self._metas]

    # ---------------------------
    # PUBLIC API used by MainWindow
    # ---------------------------
    def ui_combo_items(self) -> List[dict]:
        """
        Elements to populate a combo box:
         - key: meta.name
         - label: meta.name
         - enabled: whether requirements are satisfied
         - tooltip, color, order
         - plugin_obj: already created instance (if it exists)
        """
        items: List[dict] = []
        for meta in sorted(self._metas, key=lambda m: getattr(m, "order", 9999)):
            _ok, missing = self._dependency_status(meta)  # <-- uses dependency_status
            ok = (len(missing) == 0)
            items.append({
                "key": meta.name,
                "label": getattr(meta, "name", "plugin"),
                "enabled": ok,
                "tooltip": f"{meta.name} {meta.version or ''}".strip() + ("" if ok else f" (missing: {', '.join(missing)})"),
                "color": "green" if ok else "gray",
                "order": getattr(meta, "order", 9999),
                "plugin_obj": self._instances.get(meta.name),
            })
        return items

    def get(self, key: str):
        """
        Retrieve a plugin instance by its key. If the plugin is not already instantiated,
        it will attempt to load and instantiate it.

        Args:
            key: The unique name of the plugin.

        Returns:
            The plugin instance if successful, otherwise None.
        """
        if key in self._instances:
            return self._instances[key]
        meta = self._meta_by_name(key)
        if meta is None or not self._requirements_ok(meta):
            return None
        module = self._import_plugin_package(meta)
        if module is None:
            logging.warning("Cannot import plugin module for '%s'", meta.name)
            return None
        factory_or_cls = self._resolve_entry_point(module, meta.entry_point)
        if factory_or_cls is None:
            logging.warning("No entry point found for plugin '%s'", meta.name)
            return None
        try:
            obj = factory_or_cls(self.parent)
        except TypeError:
            obj = factory_or_cls()
        except Exception as exc:
            logging.error("Failed to instantiate plugin '%s': %s", meta.name, exc)
            return None
        self._instances[key] = obj
        return obj

    # Alias for compatibility
    def plugin_by_key(self, key: str):
        """
        Alias for the `get` method to retrieve a plugin instance by its key.
        """
        return self.get(key)

    def run(self, key: str, **context) -> bool:
        """
        Execute a plugin if it exposes one of the following methods:
        run, exec, execute, show, or __call__.

        Args:
            key: The unique name of the plugin.
            **context: Additional context to pass to the plugin's method.

        Returns:
            True if the plugin was successfully executed, otherwise False.
        """
        obj = self.get(key)
        if obj is None:
            return False
        for method in ("run", "exec", "execute", "show", "__call__"):
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn(**context)
                except TypeError:
                    try:
                        fn(self.parent)
                    except Exception:
                        fn()
                return True
        return False

    # ---------------------------
    # INTERNAL HELPERS for the instance
    # ---------------------------
    def _instantiate_from_meta(self, meta: PluginMeta):
        """
        Import the plugin package, resolve the entry point in the imported module
        (using _resolve_entry_point(module, entry_point)), and instantiate the object.

        Args:
            meta: The metadata of the plugin to instantiate.

        Returns:
            The instantiated plugin object, or None if instantiation fails.
        """
        # Ensure the plugin folder is importable
        if meta.path:
            p = str(meta.path.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        module = self._import_plugin_package(meta)
        if module is None:
            logging.warning("Cannot import plugin module for '%s'", meta.name)
            return None

        # Use the _resolve_entry_point method to find the factory/class
        factory_or_cls = self._resolve_entry_point(module, meta.entry_point)
        if factory_or_cls is None:
            logging.warning("No entry point found for plugin '%s'", meta.name)
            return None

        # Robust instantiation: first try factory(parent), then classes with various signatures
        try:
            return factory_or_cls(self.parent)
        except TypeError:
            # Try some common signatures
            try:
                return factory_or_cls(parent=self.parent, manager=self)
            except TypeError:
                try:
                    return factory_or_cls(self.parent, self)
                except TypeError:
                    try:
                        return factory_or_cls(parent=self.parent)
                    except TypeError:
                        try:
                            return factory_or_cls(manager=self)
                        except TypeError:
                            try:
                                return factory_or_cls()
                            except Exception as exc:
                                logging.error("Failed to instantiate plugin '%s': %s", meta.name, exc)
                                return None
        except Exception as exc:
            logging.error("Failed to instantiate plugin '%s': %s", meta.name, exc)
            return None