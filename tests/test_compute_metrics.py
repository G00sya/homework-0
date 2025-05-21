import os
import json
import pytest
from compute_metrics import compute_metrics  # Убедитесь, что путь к модулю правильный
from pathlib import Path


def test_compute_metrics():
    path = Path("final_metrics.json")
    if path.exists():
        os.remove(path)

    compute_metrics()
    assert path.exists(), "Файл final_metrics.json не был создан"

    with open(path, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            pytest.fail("Файл final_metrics.json содержит невалидный JSON")

    assert "accuracy" in metrics, "В файле final_metrics.json отсутствует ключ 'accuracy'"
    assert isinstance(metrics["accuracy"], (int, float)), "Значение accuracy должно быть числом (int или float)"

    accuracy = metrics["accuracy"]
    assert 0 <= accuracy <= 1, f"Значение accuracy ({accuracy}) должно быть в диапазоне от 0 до 1"
