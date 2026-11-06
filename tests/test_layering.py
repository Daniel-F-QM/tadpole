"""The import rules from ARCHITECTURE.md, enforced (Lillypad does the same)."""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "tadpole"


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def files(sub):
    return sorted((PKG / sub).rglob("*.py"))


def test_design_never_imports_gui():
    for f in files("design"):
        assert not any(m.startswith("tadpole.gui") for m in imports_of(f)), f


def test_core_layers_have_no_qt():
    for sub in ("prep", "io", "engine"):
        for f in files(sub):
            bad = [m for m in imports_of(f) if m.startswith(("PySide6", "matplotlib"))]
            assert not bad, (f, bad)
    for name in ("analysis.py", "paths.py", "settings.py"):
        assert not any(m.startswith("PySide6") for m in imports_of(PKG / name))


def test_only_the_worker_imports_juliacall():
    for f in PKG.rglob("*.py"):
        if any(m.startswith("juliacall") for m in imports_of(f)):
            assert f.name == "julia_worker.py", f


def test_stylesheet_tokens_all_defined(qapp):
    from tadpole.design.stylesheet import stylesheet_tokens
    from tadpole.design.tokens import DARK_PALETTE, LIGHT_PALETTE, METRICS, TYPE
    known = set(DARK_PALETTE) | set(TYPE) | set(METRICS) | {"arrow_up", "arrow_dn", "check"}
    assert stylesheet_tokens() <= known
    assert set(DARK_PALETTE) == set(LIGHT_PALETTE)
