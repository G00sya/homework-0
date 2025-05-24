import os
import json
import pytest
from src.compute_metrics import compute_metrics
from pathlib import Path


@pytest.fixture
def clean_metrics_file():
    path = Path(__file__).parent / "final_metrics.json"
    if path.exists():
        path.unlink()
    yield
    if path.exists():
        path.unlink()


def test_compute_metrics(clean_metrics_file):
    compute_metrics()

    path = Path(__file__).parent / "final_metrics.json"
    with open(path, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            pytest.fail("Файл final_metrics.json содержит невалидный JSON")

    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], (int, float))
    assert 0 <= metrics["accuracy"] <= 1