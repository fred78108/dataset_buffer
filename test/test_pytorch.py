"""Unit tests for the DatasetBuffer

python -m pytest -s test/test_pytorch.py
"""

import torch
from dataset_buffer.buffer import DatasetBuffer


def test_normalize_torch_tensor_returns_list():
    buffer = DatasetBuffer()
    sample = torch.tensor([[1, 2], [3, 4]])
    normalized = buffer.normalize(sample)
    assert normalized["tensor"] == sample.cpu().numpy().tolist()
    assert isinstance(normalized["tensor"], list)


def test_add_torch_tensor_samples_and_to_pydict():
    buffer = DatasetBuffer()
    samples = [torch.tensor([i, i + 1]) for i in range(2)]
    buffer.append(samples)
    assert buffer.to_pydict()["tensor"] == [[0, 1], [1, 2]]


def test_to_torch_converts_numeric_columns_to_tensors():
    buffer = DatasetBuffer()
    samples = [{"int_value": i, "float_value": i + 0.5} for i in range(3)]
    buffer.append(samples)
    tensors = buffer.to_torch()
    assert isinstance(tensors["int_value"], torch.Tensor)
    assert isinstance(tensors["float_value"], torch.Tensor)
    assert torch.equal(tensors["int_value"], torch.tensor([0, 1, 2]))
    expected_float = torch.tensor(
        [0.5, 1.5, 2.5], dtype=tensors["float_value"].dtype)
    assert torch.allclose(tensors["float_value"], expected_float)


def test_get_batch_with_valid_indices_returns_expected_rows():
    buffer = DatasetBuffer()
    samples = [{"key": i} for i in range(5)]
    buffer.append(samples)
    batch = buffer.get_batch([0, 2, 4]).to_pydict()
    assert batch["key"] == [0, 2, 4]


def test_add_with_missing_keys_aligns_schema():
    buffer = DatasetBuffer()
    buffer.append([{"x": 1}, {"y": 2}])
    data = buffer.to_pydict()
    assert data["x"] == [1, None]
    assert data["y"] == [None, 2]
