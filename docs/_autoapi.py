# (code_tree generator moved to docs/_code_tree.py)
# docs/_autoapi.py (filesystem-based, namespace-safe)
import pathlib, sys
import mkdocs_gen_files

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = "cloud2fem"
PKG_ROOT = ROOT / PKG

# Exclude specific subpackages from Auto-API generation (they may be optional or external)
EXCLUDE_TOP_DIRS = {"plugins"}  # skip cloud2fem/plugins/**
EXCLUDE_MODULE_PREFIX = (f"{PKG}.plugins.",)

# Ensure package is importable for mkdocstrings
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

def module_name_for(py: pathlib.Path) -> str | None:
    """Return dotted module for a Python file under PKG_ROOT, or None for dunders or excluded dirs.
    Example: cloud2fem/ui/main_window.py -> "cloud2fem.ui.main_window".
    Skips: __init__.py, __main__.py, and files under `EXCLUDE_TOP_DIRS`.
    """
    if py.name in {"__init__.py", "__main__.py"}:
        return None
    rel = py.relative_to(PKG_ROOT)  # e.g. ui/main_window.py
    # Skip files under excluded top-level dirs (e.g., plugins/**)
    if rel.parts and rel.parts[0] in EXCLUDE_TOP_DIRS:
        return None
    dotted = ".".join(rel.with_suffix("").parts)
    return f"{PKG}.{dotted}"

# Collect all modules by walking the filesystem (works for namespace packages)
modules = []
for py in PKG_ROOT.rglob("*.py"):
    mod = module_name_for(py)
    if mod and not mod.startswith(EXCLUDE_MODULE_PREFIX):
        modules.append(mod)

modules = sorted(set(modules))

# Generate one markdown per module
for mod in modules:
    out_path = pathlib.Path("api") / f"{mod}.md"
    with mkdocs_gen_files.open(out_path, "w") as fd:
        fd.write(f"# `{mod}`\n\n::: {mod}\n    options:\n      members_order: source\n      show_source: true\n")
    # Configure the edit path so "view source" can link to the file in the repo
    src_rel = mod.replace(".", "/") + ".py"
    mkdocs_gen_files.set_edit_path(out_path, src_rel)

# Build API index
with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write("# API Index\n\n")
    for mod in modules:
        fd.write(f"- [{mod}](./{mod}.md)\n")