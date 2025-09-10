# tests/test_plugin.py
# Esegue l'import delle 3 nuvole di esempio usando l'helper import_cloud(path).
# Executes the import of the 3 example clouds using the helper import_cloud(path).
# Disponibili nel contesto: window, import_cloud, np, pv, mct, mcts, QtWidgets,...
# Available in the context: window, import_cloud, np, pv, mct, mcts, QtWidgets,...

import os
import pytest

pytest.skip("Requires application context", allow_module_level=True)

project_root = os.path.dirname(os.path.dirname(__file__))
tests_dir = os.path.join(project_root, "tests")

files = [
    "test_1_Corinthian_Column_Capital_RGB_no_normals.ply",
    "test_2_Rocca_North_tower_no_RGB.ply",
    "test_3_Turkish_pillar_RGB_normals.ply",
]

for name in files:
    p = os.path.join(tests_dir, name)
    if os.path.isfile(p):
        # axis_preset: "Z-up (identità)" | "Y-up (scambia Y/Z)" | "X-up (scambia X/Z)" | "Flip X/Y/Z"
        # axis_preset: "Z-up (identity)" | "Y-up (swap Y/Z)" | "X-up (swap X/Z)" | "Flip X/Y/Z"
        # color_preference: "auto" (prefer RGB se disponibile), "rgb", "colormap"
        # color_preference: "auto" (prefer RGB if available), "rgb", "colormap"
        import_cloud(
            p,
            axis_preset="Z-up (identity)",
            color_preference="auto",
            compute_normals_if_missing=True,
            map_normals=True,
        )
    else:
        print(f"[WARN] Missing test file: {name}")
