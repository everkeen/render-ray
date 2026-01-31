"""Render Ray Constants"""

import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = PACKAGE_ROOT.parent
if ROOT_DIR.name not in ("rr", "render_ray", "RenderRay"):
    ROOT_DIR = PACKAGE_ROOT  # Packaged installation
LOGS_DIR = ROOT_DIR / "logs"
MESHES_DIR = ROOT_DIR / "meshes"
