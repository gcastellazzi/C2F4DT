# tests/test_import_ply.py
# Imports all .ply files in the bunny_data folder using import_cloud(path).
# Available in the context: window, import_cloud, np, pv, mct, mcts, QtWidgets,...

import os
# import pytest

# pytest.skip("Requires application context", allow_module_level=True)

project_root = os.path.dirname(os.path.dirname(__file__))
tests_dir = os.path.join(project_root, "tests", "bunny_data")

if not os.path.isdir(tests_dir):
    print(f"[WARN] bunny_data folder not found: {tests_dir}")

ply_files = [f for f in os.listdir(tests_dir) if f.lower().endswith(".ply")]

for name in sorted(ply_files):
    p = os.path.join(tests_dir, name)
    if os.path.isfile(p):
        # axis_preset: "Z-up (identity)" | "Y-up (swap Y/Z)" | "X-up (swap X/Z)" | "Flip X/Y/Z"
        # color_preference: "auto" (prefer RGB if available), "rgb", "colormap"
        import_cloud(
            p,
            axis_preset="Z-up (identity)",
            color_preference="auto",
            compute_normals_if_missing=True,
            map_normals=True,
        )
        print(f"[INFO] Imported {name}")
    else:
        print(f"[WARN] Missing test file: {name}")