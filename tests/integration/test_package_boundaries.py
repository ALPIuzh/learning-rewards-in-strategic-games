"""Dependency boundaries between the library, workbench, and applications."""

import ast

import pytest


@pytest.mark.parametrize(
    ("package_name", "forbidden_roots"),
    [("mairl", {"hydra", "mairl_exp", "matplotlib", "omegaconf", "experiments"}), ("mairl_exp", {"experiments"})],
)
def test_package_dependencies(repository_root, package_name, forbidden_roots):
    package = repository_root / "src" / package_name
    violations = []
    for source in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            else:
                continue
            for module in imported:
                if module.split(".", maxsplit=1)[0] in forbidden_roots:
                    relative = source.relative_to(package.parent)
                    violations.append(f"{relative}:{node.lineno} imports {module}")
    assert not violations, "forbidden dependencies:\n" + "\n".join(violations)
