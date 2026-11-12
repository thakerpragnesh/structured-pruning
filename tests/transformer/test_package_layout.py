"""The package's folder layout is an architecture, so it is tested like one.

`prunelib.transformer/` is grouped by role, and each subpackage may depend
only on the subpackages listed before it in `LAYERS`. That is what keeps the
grouping meaningful: without a check, a convenient import from `surgery`
into `pipeline` (or from `models` into anything) would quietly turn the
folders back into one tangled namespace, with cycles waiting to happen.
These tests parse the source rather than importing it, so a violation is
reported as the offending import line, not as a circular-import traceback.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "prunelib"
PACKAGE = ROOT / "transformer"

# Lowest layer first. A subpackage may import from any subpackage before it.
# The core `prunelib` modules (saliency, scanners, surgery, ...) sit below all
# of these: any layer may import them, and they must never import back up.
LAYERS = ["models", "measurement", "analysis", "selection", "surgery", "pipeline"]
CORE = "<core prunelib>"


def modules():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def subpackage_of(path: Path):
    parts = path.relative_to(PACKAGE).parts
    return parts[0] if len(parts) > 1 else None


def _absolute_targets(path: Path):
    """Yield `(lineno, dotted_parts)` for every `prunelib` import in `path`,
    relative imports resolved against the file's own package."""
    package_parts = ["prunelib", *path.relative_to(ROOT).with_suffix("").parts[:-1]]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package_parts[: len(package_parts) - (node.level - 1)]
            yield node.lineno, base + (node.module.split(".") if node.module else [])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "prunelib":
            yield node.lineno, node.module.split(".")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "prunelib":
                    yield node.lineno, alias.name.split(".")


def internal_imports(path: Path):
    """Yield `(lineno, target)` for every import of the package from inside
    `path` -- relative or absolute, `TYPE_CHECKING` or not. `target` is a
    subpackage name, `None` for the top-level `prunelib.transformer`
    namespace itself, or `CORE` for a core `prunelib` module."""
    for lineno, parts in _absolute_targets(path):
        if parts[:2] == ["prunelib", "transformer"]:
            yield lineno, (parts[2] if len(parts) > 2 else None)
        elif len(parts) > 1:
            yield lineno, CORE
        else:
            yield lineno, None  # bare `prunelib` re-imports the namespace that imports us


def test_core_never_imports_the_transformer_subpackage():
    """Core `prunelib` is the lowest layer: CNN users must not pay for (or
    cycle through) the transformer stack."""
    violations = [
        f"{p.relative_to(ROOT)}:{lineno}"
        for p in sorted(ROOT.glob("*.py"))
        for lineno, parts in _absolute_targets(p)
        if parts[:2] == ["prunelib", "transformer"]
    ]
    assert not violations, violations


def test_every_module_lives_in_a_known_subpackage():
    stray = [
        str(p.relative_to(PACKAGE)) for p in modules()
        if p.name != "__init__.py" and subpackage_of(p) not in LAYERS
    ]
    assert not stray, (
        f"Modules outside the layered subpackages: {stray}. Put new code in one of "
        f"{LAYERS} (or add a layer to LAYERS and the package docstring)."
    )


def test_every_subpackage_is_a_package():
    for layer in LAYERS:
        assert (PACKAGE / layer / "__init__.py").is_file(), f"{layer}/ has no __init__.py"


def test_module_names_are_unique_across_subpackages():
    """Docs and docstrings refer to modules by bare name (`head_analysis`,
    `ffn_surgery`); that only stays unambiguous while names are unique."""
    names = [p.stem for p in modules() if p.name != "__init__.py"]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)


def test_subpackages_only_depend_on_lower_layers():
    violations = []
    for path in modules():
        layer = subpackage_of(path)
        if layer is None:
            continue  # the top-level __init__ re-exports everything, by design
        rank = LAYERS.index(layer)
        for lineno, target in internal_imports(path):
            where = f"{path.relative_to(PACKAGE)}:{lineno}"
            if target is None:
                violations.append(f"{where} imports the top-level package (a cycle)")
            elif target == CORE:
                continue  # core prunelib is below every layer
            elif target != layer and (target not in LAYERS or LAYERS.index(target) > rank):
                violations.append(f"{where} ({layer}) imports {target}, a higher layer")
    assert not violations, "\n".join(violations)
