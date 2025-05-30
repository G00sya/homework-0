from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm, trange
import wandb
from typing import Tuple, Dict

from configs.hparams import config
from src.prepare_data import prepare_data


def config_train_process(
    train_dataset: Dataset, test_dataset: Dataset, device: torch.device
) -> Tuple[DataLoader, DataLoader, nn.Module, nn.Module, optim.Optimizer]:
    """
    Config several objects for training process: data loaders, model, loss function and optimizer.

    :param train_dataset: Dataset for train.
    :param test_dataset: Dataset for test.
    :param device: Device: "cpu", "cuda".
    :return: Data loaders, model, loss function and optimizer.
    """
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"]
    )

    model = resnet18(
        pretrained=False,
        num_classes=config["num_classes"],
        zero_init_residual=config["zero_init_residual"],
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    return train_loader, test_loader, model, criterion, optimizer


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy which equals part of right answers.

    :param preds: Predictions of a model.
    :param targets: Right answers.
    :return: Accuracy as float.
    """
    result = (targets == preds).float().mean()
    return result


def estimate_current_state(
    test_loader: DataLoader,
    device: torch.device,
    model: nn.Module,
    loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Calculate metrics for current state of training.

    :param test_loader: Data loader for test data.
    :param device: Device: "cpu", "cuda".
    :param model: Training model.
    :param loss: Loss tensor for current state.
    :return: Dictionary for wandb: metrics = {'test_acc': accuracy, 'train_loss': loss}.
    """
    all_preds = []
    all_labels = []

    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        with torch.inference_mode():
            outputs = model(test_images)
            preds = torch.argmax(outputs, 1)

            all_preds.append(preds)
            all_labels.append(test_labels)

    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))

    metrics = {"test_acc": accuracy, "train_loss": loss}
    return metrics


def train_one_epoch(
    images: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Train one epoch. Part of the train cycle.

    :param images: Images for predictions.
    :param labels: Right labels for each image from images.
    :param model: Training model.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param device: Device: "cpu", "cuda".
    :return: Loss tensor.
    """
    if criterion is None:
        raise AttributeError("Criterion cannot be None - cannot compute loss")
    if optimizer is None:
        raise AttributeError("Optimizer cannot be None - cannot update weights")

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def save_model(model: nn.Module, path="weights/model.pt") -> None:
    """
    Save weights of trained model and wandb info.

    :param model: Trained model.
    :param path: Path with a name to save weights.
    :return: None.
    """
    torch.save(model.state_dict(), path)

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)


def train_model(
    train_dataset: Dataset,
    test_dataset: Dataset,
    device_name="cuda",
    path="weights/model.pt",
    run="release",
) -> None:
    """
    Main function for model training.

    :param train_dataset: Dataset for train.
    :param test_dataset: Dataset for test.
    :param device_name: Name of device for calculation: "cuda" or "cpu".
    :param path: Path to save model's weights and wandb info.
    :param run: Name of wandb session.
    :return: None.
    """
    wandb.init(config=config, project="ML Homework-0", name=run)
    # if run == "test":
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    train_loader, test_loader, model, criterion, optimizer = config_train_process(
        train_dataset, test_dataset, device
    )
    wandb.watch(model)

    for epoch in trange(config["epochs"]):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            loss = train_one_epoch(images, labels, model, criterion, optimizer, device)

            if i % 100 == 0:
                metrics = estimate_current_state(test_loader, device, model, loss)
                wandb.log(
                    metrics,
                    step=epoch * len(train_dataset) + (i + 1) * config["batch_size"],
                )

    save_model(model, path)
    wandb.finish()


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_data()
    train_model(train_dataset, test_dataset)
