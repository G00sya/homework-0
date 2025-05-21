from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from typing import Tuple

import torchvision.transforms as transforms


def download_data() -> None:
    """
    Скачивает данные CIFAR10 из библиотеки torchvision и сохраняет их
    :return: None.
    """
    CIFAR10("data/CIFAR10/train", download=True)
    CIFAR10("data/CIFAR10/test", download=True)
    return


def prepare_data(path=str(Path(__file__).parent / "data" / "CIFAR10")) -> Tuple[Dataset, Dataset]:
    """
    Инициализирует обучающие и тестовые данные CIFAR10 с нормализованными тензорами.

    :param path: Путь к папке с данными CIFAR10
    :return: Кортеж с обучающим и тестовым датасетами torchvision.datasets.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    print(path)
    train_dataset = CIFAR10(
        root=path + "/train",
        train=True,
        transform=transform,
        download=False
    )

    test_dataset = CIFAR10(
        root=path + "/test",
        train=False,
        transform=transform,
        download=False,
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    download_data()
