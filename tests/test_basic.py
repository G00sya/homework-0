import pytest
import torch


def test_arange_elems() -> None:
    arr = torch.arange(0, 10, dtype=torch.float32)
    assert torch.allclose(arr[-1], torch.tensor([9.0]))


def test_div_zero() -> None:
    a = torch.zeros(1, dtype=torch.long)
    b = torch.ones(1, dtype=torch.long)

    assert torch.isinf(b / a)


def test_div_zero_python() -> None:
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_tensor_creation() -> None:
    tensor = torch.randn(2, 3, dtype=torch.float64)
    assert tensor.size() == torch.Size([2, 3])
    assert tensor.dtype == torch.float64


def test_tensor_addition() -> None:
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    result = a + b
    expected = torch.tensor([5, 7, 9], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_matmul() -> None:
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    result = torch.matmul(a, b)
    assert result.size() == torch.Size([2, 4])


def test_tensor_indexing() -> None:
    tensor = torch.arange(12).reshape(3, 4)
    assert tensor[0, 0] == 0
    assert tensor[1, 2] == 6
    assert tensor[2, 3] == 11


def test_tensor_cuda() -> None:
    if torch.cuda.is_available():
        tensor = torch.randn(2, 3).cuda()
        assert tensor.is_cuda
    else:
        pytest.skip("CUDA is not available")


def test_tensor_mean() -> None:
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    mean = torch.mean(tensor)
    assert torch.allclose(mean, torch.tensor(2.5))


def test_tensor_reshape() -> None:
    tensor = torch.arange(12)
    reshaped_tensor = tensor.reshape(3, 4)
    assert reshaped_tensor.size() == torch.Size([3, 4])
    assert reshaped_tensor.numel() == 12
