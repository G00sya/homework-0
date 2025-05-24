import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from typing import Tuple
import os
from pathlib import Path

from unittest.mock import Mock, patch

from src.prepare_data import download_data, prepare_data
from src import train


@pytest.fixture
def prepare_dataset() -> Tuple[Dataset, Dataset]:
    download_data()
    return prepare_data()


@pytest.fixture
def prepare_one_epoch_train():
    images = Mock()
    images.to = Mock()

    labels = Mock()
    labels.to = Mock()

    # Create mock analogs
    mock_model = Mock(return_value=labels)
    mock_param_1 = nn.Parameter(torch.randn(5, 5))
    mock_param_2 = nn.Parameter(torch.randn(10))
    mock_parameters = [mock_param_1, mock_param_2]
    mock_model.parameters = Mock(return_value=mock_parameters)

    mock_criterion = Mock()
    mock_criterion.return_value = Mock()
    mock_criterion.return_value.backward = Mock()

    mock_optimizer = optim.SGD(mock_model.parameters(), lr=0.01)
    mock_optimizer.zero_grad = Mock(side_effect=mock_optimizer.zero_grad)
    mock_optimizer.step = Mock()
    mock_optimizer.zero_grad = Mock()

    return images, labels, mock_model, mock_criterion, mock_optimizer


@pytest.mark.parametrize(["device_name"], [["cpu"], ["cuda"]])
def test_train_one_epoch(device_name, prepare_one_epoch_train):
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    images, labels, model, criterion, optimizer = prepare_one_epoch_train
    loss = train.train_one_epoch(images, labels, model, criterion, optimizer, device)

    # Check if data was transferred to the device
    images.to.assert_called_with(device), f"Images weren't transferred to {device_name}"
    labels.to.assert_called_with(device), f"Labels weren't transferred to {device_name}"

    # Change mocks for next tests
    images = images.to(device)
    labels = labels.to(device)

    # Check model calls
    model.assert_called_once(), "Model wasn't called once"
    model.assert_called_with(images), "Model got wrong input"

    # Check loss backward call
    (
        criterion.assert_called_with(model.return_value, labels),
        "Criterion got wrong args, must be model_output, labels",
    )
    (
        criterion.return_value.backward.assert_called_once(),
        "loss.backward() wasn't called",
    )

    # Check optimizer calls
    optimizer.step.assert_called_once(), "optimizer.step() wasn't called"
    optimizer.zero_grad.assert_called_once(), "optimizer.zero_grad() wasn't called"

    assert loss == criterion.return_value, "Wrong output of train function"


@pytest.mark.parametrize(["device_name"], [["invalid_device"]])
def test_train_one_epoch_device_error(device_name, prepare_one_epoch_train):
    with pytest.raises(Exception):
        device = torch.device(device_name)
        images, labels, model, criterion, optimizer = prepare_one_epoch_train
        train.train_one_epoch(images, labels, model, criterion, optimizer, device)


def test_train_one_epoch_input_types(prepare_one_epoch_train):
    images, labels, model, criterion, optimizer = prepare_one_epoch_train
    device = torch.device("cpu")

    # Try with incorrect types
    with pytest.raises((AttributeError, TypeError)):
        train.train_one_epoch([1, 2, 3], labels, model, criterion, optimizer, device)

    with pytest.raises((AttributeError, TypeError)):
        train.train_one_epoch(images, [1, 2, 3], model, criterion, optimizer, device)


def test_train_one_epoch_no_parameters(prepare_one_epoch_train):
    images, labels, model, criterion, optimizer = prepare_one_epoch_train

    model.parameters.return_value = []

    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()

    device = torch.device("cpu")

    with patch('src.train.optim.SGD', return_value=optimizer):
        loss = train.train_one_epoch(images, labels, model, criterion, optimizer, device)

    optimizer.zero_grad.assert_called_once()
    assert loss == criterion.return_value


def test_train_one_epoch_none_criterion_optimizer(prepare_one_epoch_train):
    images, labels, model, criterion, optimizer = prepare_one_epoch_train
    device = torch.device("cpu")

    with pytest.raises(AttributeError):
        train.train_one_epoch(images, labels, model, None, optimizer, device)

    with pytest.raises(AttributeError):
        train.train_one_epoch(images, labels, model, criterion, None, device)


def test_compute_accuracy():
    preds = torch.randint(0, 2, size=(100,))
    targets = preds.clone()

    assert train.compute_accuracy(preds, targets) == 1.0

    preds = torch.tensor([1, 2, 3, 0, 0, 0])
    targets = torch.tensor([1, 2, 3, 4, 5, 6])

    assert train.compute_accuracy(preds, targets) == 0.5


def compute_accuracy(preds, targets):
    """Computes accuracy for given predictions and targets."""
    if len(preds) == 0:
        return torch.tensor(float('nan'))  # Возвращаем NaN, если тензоры пустые

    correct = (preds == targets).sum().item()
    total = len(targets)
    return correct / total


def test_compute_accuracy_empty_tensors():
    preds = torch.tensor([])
    targets = torch.tensor([])
    assert torch.isnan(train.compute_accuracy(preds, targets))


def test_compute_accuracy_different_data_types():
    preds = torch.tensor([1, 2, 3], dtype=torch.int64)
    targets = torch.tensor([1, 2, 3], dtype=torch.float32)

    accuracy = train.compute_accuracy(preds, targets)
    assert accuracy == 1.0


def test_compute_accuracy_different_sizes():
    preds = torch.tensor([1, 2, 3])
    targets = torch.tensor([1, 2, 3, 4])

    with pytest.raises(RuntimeError):
        train.compute_accuracy(preds, targets)


@pytest.mark.parametrize(
    "preds,targets,result",
    [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 1.0),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 0]), 0.0),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0]), 2 / 3),
    ],
)
def test_compute_accuracy_parametrized(preds, targets, result):
    assert torch.allclose(
        train.compute_accuracy(preds, targets),
        torch.tensor([result]),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize(["device_name"], [["cpu"], ["cuda"]])
def test_estimate_current_state_validity(device_name, prepare_dataset):
    # Prepare estimation
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    loss = torch.tensor(0.5)

    train_dataset, test_dataset = prepare_dataset
    train_loader, test_loader, model, criterion, optimizer = train.config_train_process(
        train_dataset, test_dataset, device
    )

    # Estimate
    metrics = train.estimate_current_state(test_loader, device, model, loss)

    # Check validity
    assert "test_acc" in metrics, "test_acc is not in metrics"
    assert "train_loss" in metrics, "train_loss is not in metrics"
    assert isinstance(
        metrics["test_acc"], torch.Tensor
    ), "test_acc is not a torch.Tensor"
    assert isinstance(
        metrics["train_loss"], torch.Tensor
    ), "train_loss is not a torch.Tensor"
    assert metrics["train_loss"] == loss, "Loss was changed inside the function"


def test_estimate_current_state_empty_test_loader(device_name, prepare_dataset):
    # Prepare estimation
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    loss = torch.tensor(0.5)

    train_dataset, test_dataset = prepare_dataset
    train_loader, test_loader, model, criterion, optimizer = train.config_train_process(
        train_dataset, test_dataset, device
    )

    # Mock the test_loader to return an empty iterator
    test_loader = []

    # Estimate
    metrics = train.estimate_current_state(test_loader, device, model, loss)

    # Check validity
    assert "test_acc" in metrics, "test_acc is not in metrics"
    assert "train_loss" in metrics, "train_loss is not in metrics"
    assert isinstance(
        metrics["test_acc"], torch.Tensor
    ), "test_acc is not a torch.Tensor"
    assert isinstance(
        metrics["train_loss"], torch.Tensor
    ), "train_loss is not a torch.Tensor"
    assert metrics["train_loss"] == loss, "Loss was changed inside the function"
    assert torch.isnan(metrics["test_acc"])


@pytest.mark.parametrize(["device_name"], [["cuda"]])
def test_estimate_current_state_no_cuda(device_name, prepare_dataset):
    if not torch.cuda.is_available():
        pytest.skip("CUDA не доступна, пропускаем тест.")

    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    loss = torch.tensor(0.5)

    train_dataset, test_dataset = prepare_dataset
    train_loader, test_loader, model, criterion, optimizer = train.config_train_process(
        train_dataset, test_dataset, device
    )
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError):
            train.estimate_current_state(test_loader, device, model, loss)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


def generate_dummy_model():
    # Create a mock model for testing purposes
    model = MockModel()
    model.linear.weight.data.fill_(0.5)  # Set some specific weights
    model.linear.bias.data.fill_(0.25)
    return model


def test_save_model():
    model = generate_dummy_model()
    path = Path(__file__).parent / "weights" / "model.pt"  # Path to save the model weights

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Mock wandb to avoid actual wandb calls during testing
    with patch("wandb.run") as mock_wandb_run:
        mock_wandb_run.id = "mock_run_id"
        mock_wandb_run.dir = "mock_dir"

        train.save_model(model, path)

        # 1. Check if the path exists
        assert os.path.exists(path), "File doesn't exists"

        # 2. Load the saved weights and compare them to the original weights
        loaded_model = generate_dummy_model()
        loaded_model.load_state_dict(torch.load(path))

        # Check similarity of weights
        assert torch.allclose(model.linear.weight.data, loaded_model.linear.weight.data)
        assert torch.allclose(model.linear.bias.data, loaded_model.linear.bias.data)

        # 3. Verify that the run_id.txt file was created and contains the correct run ID
        run_id_path = "run_id.txt"
        assert os.path.exists(run_id_path)

        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
        assert run_id == "mock_run_id"

        os.remove(run_id_path)


@pytest.mark.parametrize(["device_name"], [["cuda"]])
def test_training(device_name, prepare_dataset):
    train_dataset, test_dataset = prepare_dataset
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    train_loader, test_loader, model, criterion, optimizer = train.config_train_process(
        train_dataset, test_dataset, device
    )
    path = Path(__file__).parent / "weights" / "model.pt"  # Path to save the model weights
    train.train_model(train_dataset, test_dataset, device_name, path, "test")

    model.load_state_dict(torch.load(path))
    model.to(device)

    try:
        img_size = 32
        model(torch.rand(32, 3, img_size, img_size).to(device))
    except Exception as e:
        pytest.fail(f"Model loading failed with exception: {e}")
