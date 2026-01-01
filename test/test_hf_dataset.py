"""Unit tests for the DatasetBuffer

python -m pytest -s test/test_hf_dataset.py
"""

from typing import cast
import numpy as np
import pytest
import torch
from datasets import Dataset, Image
from dataset_buffer.buffer import DatasetBuffer
import PIL.Image as pil_image


def test_add_torch_tensor_flattened():
    buffer = DatasetBuffer()
    sample = torch.tensor([4, 5, 6])
    buffer.append(sample)
    assert len(buffer) == 1
    # Adjusted to handle nested structure
    assert buffer[0]["tensor"] == [[4, 5, 6]]


def test_add_multiple_torch_tensors():
    buffer = DatasetBuffer()
    samples = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
    buffer.append(samples)
    assert len(buffer) == 2
    assert buffer[0]["tensor"] == [[7, 8, 9]]
    assert buffer[1]["tensor"] == [[10, 11, 12]]


def test_add_huggingface_dataset():
    buffer = DatasetBuffer()
    hf_dataset = Dataset.from_dict({"col1": [1, 2], "col2": ["a", "b"]})
    buffer.append(hf_dataset[0])
    assert len(buffer) == 1


def test_get_batch():
    buffer = DatasetBuffer()
    samples = [{"key": i} for i in range(5)]
    buffer.append(samples)
    batch = buffer.get_batch([1, 3])
    assert len(batch) == 2


def test_normalize_unsupported_type():
    buffer = DatasetBuffer()
    with pytest.raises(TypeError):
        buffer.append(object())


def test_contains():
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    assert sample in buffer


def test_len_empty():
    buffer = DatasetBuffer()
    assert len(buffer) == 0


def test_len_non_empty():
    buffer = DatasetBuffer()
    buffer.append({"key": "value"})
    assert len(buffer) == 1


def test_add_empty_list():
    buffer = DatasetBuffer()
    buffer.append([])
    assert len(buffer) == 0


def test_add_none():
    buffer = DatasetBuffer()
    with pytest.raises(TypeError):
        buffer.append(None)


def test_add_large_torch_tensor():
    buffer = DatasetBuffer()
    large_tensor = torch.rand(1000, 1000)
    buffer.append(large_tensor)
    assert len(buffer) == 1
    assert np.allclose(buffer[0]["tensor"], large_tensor.numpy().tolist())


def test_add_image():
    buffer = DatasetBuffer()
    arr = (np.random.rand(128, 128, 3) * 255).astype("uint8")
    image = pil_image.fromarray(arr)
    image = cast(Image, image)
    buffer.append(image)
    assert len(buffer) == 1
    assert "image" in buffer[0]  # normalizer adds "image"


def test_add_duplicate_items():
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    buffer.append(sample)
    assert len(buffer) == 2


def test_add_and_check_contains():
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    assert sample in buffer


def test_add_and_get_batch_with_invalid_indices():
    buffer = DatasetBuffer()
    samples = [{"key": i} for i in range(5)]
    buffer.append(samples)
    with pytest.raises(IndexError):
        buffer.get_batch([10, 20])


def test_to_hf_dataset():
    buffer = DatasetBuffer()
    samples = [{"key": i} for i in range(5)]
    buffer.append(samples)
    result = buffer.to_hf_dataset()
    assert isinstance(result, Dataset)


def test_text_structured_dataset():
    buffer = DatasetBuffer()
    samples = [
        {
            "left": {
                "a": "a",
                "b": 123
            },
            "right": {
                "a": "b",
                "b": "444"
            },
            "label": 0
        },
        {
            "left": {
                "a": "a",
                "b": 555
            },
            "right": {
                "a": "a",
                "b": "555"
            },
            "label": 1
        }
    ]
    ds = Dataset.from_list(samples)
    # print(ds.to_iterable_dataset())
    buffer.append(ds)
    result = buffer.to_hf_dataset()
    assert isinstance(result, Dataset)
    assert result[0] in samples
    assert result[1] in samples
