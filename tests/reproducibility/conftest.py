from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from reproducibility.figures.manifest import load_manifest


@pytest.fixture(scope="session")
def figure_spec() -> Callable[[str], Any]:
    """Look up a registered figure by id, parsing the manifest once per session."""

    manifest = load_manifest()
    specs = {figure.id: figure for figure in manifest.figures}

    def lookup(figure_id: str):
        if figure_id not in specs:
            raise AssertionError(f"figure {figure_id!r} is not registered")
        return specs[figure_id]

    return lookup


@pytest.fixture
def figure_payload(figure_spec) -> Callable[[str], Any]:
    """Parse a figure's JSON descriptor.

    Function-scoped on purpose: several tests mutate the payload to assert that
    validation rejects it, so each test must get its own copy.
    """

    return lambda figure_id: json.loads(figure_spec(figure_id).data.read_text())
