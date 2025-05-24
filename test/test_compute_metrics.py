import os
import json
import pytest
from src.compute_metrics import compute_metrics
from pathlib import Path


def test_compute_metrics():
    path = Path(__file__).parent / "final_metrics.json"
    if path.exists():
        os.remove(path)

    compute_metrics()

    with open(path, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            pytest.fail("Файл final_metrics.json содержит невалидный JSON")

    assert "accuracy" in metrics, "В файле final_metrics.json отсутствует ключ 'accuracy'"
    assert isinstance(metrics["accuracy"], (int, float)), "Значение accuracy должно быть числом (int или float)"

    accuracy = metrics["accuracy"]
    assert 0 <= accuracy <= 1, f"Значение accuracy ({accuracy}) должно быть в диапазоне от 0 до 1"
