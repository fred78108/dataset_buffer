"""Unit tests for the DatasetBuffer that are not data type specific.

These tests cover the core functionality of DatasetBuffer, including
normalization, appending, retention strategies, length, containment,
batching, and error handling for unsupported types and invalid operations.

Run with:
    python -m pytest -s test/test_buffer.py
"""

import numpy as np
import pyarrow as pa
import pytest
import torch
from dataset_buffer.buffer import DatasetBuffer


def make_table(n):
    """Helper to create a simple Arrow table with one column 'x'."""
    return pa.Table.from_pylist([{"x": i} for i in range(n)])


def test_normalize_unsupported_type():
    """Test that appending an unsupported type raises TypeError."""
    buffer = DatasetBuffer()
    with pytest.raises(TypeError):
        buffer.append(object())


def test_add_dict_and_contains():
    """Test that a dict sample can be added and is contained in the buffer."""
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    assert sample in buffer


def test_add_dict_and_not_contains():
    """Test that a dict not added is not contained in the buffer."""
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    invalid_sample = {"key": "not_there"}
    assert invalid_sample not in buffer


def test_len_empty():
    """Test that a new buffer has length zero."""
    buffer = DatasetBuffer()
    assert len(buffer) == 0


def test_len_non_empty():
    """Test that buffer length increases after appending."""
    buffer = DatasetBuffer()
    buffer.append({"key": "value"})
    assert len(buffer) == 1


def test_add_empty_list():
    """Test that appending an empty list does not change buffer length."""
    buffer = DatasetBuffer()
    buffer.append([])
    assert len(buffer) == 0


def test_add_none():
    """Test that appending None raises TypeError."""
    buffer = DatasetBuffer()
    with pytest.raises(TypeError):
        buffer.append(None)


def test_add_and_check_contains():
    """Test that a sample is contained after being added."""
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    assert sample in buffer


def test_add_and_get_batch_with_invalid_indices():
    """Test that requesting a batch with invalid indices raises IndexError."""
    buffer = DatasetBuffer()
    samples = [{"key": i} for i in range(5)]
    buffer.append(samples)
    with pytest.raises(IndexError):
        buffer.get_batch([10, 20])


def test_add_diverse_types():
    """Test that appending different types raises ValueError."""
    buffer = DatasetBuffer()
    first_add = [{"key": i} for i in range(5)]
    second_add = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
    buffer.append(first_add)
    with pytest.raises(ValueError):
        buffer.append(second_add)


def test_retain_prior_zero():
    """Test that retain_prior=0 removes all rows."""
    buf = DatasetBuffer()
    buf.table = make_table(10)
    buf._apply_retain_prior(0, "random")
    assert len(buf.table) == 0


def test_retain_prior_one():
    """Test that retain_prior=1 keeps all rows."""
    buf = DatasetBuffer()
    buf.table = make_table(10)
    buf._apply_retain_prior(1, "random")
    assert len(buf.table) == 10


def test_retain_oldest():
    """Test that 'oldest' retention keeps the first k rows."""
    buf = DatasetBuffer()
    buf.table = make_table(10)
    buf._apply_retain_prior(0.3, "oldest")  # keep first 3 rows
    assert len(buf.table) == 3
    assert buf.table["x"].to_pylist() == [0, 1, 2]


def test_retain_latest():
    """Test that 'latest' retention keeps the last k rows."""
    buf = DatasetBuffer()
    buf.table = make_table(10)
    buf._apply_retain_prior(0.2, "latest")  # keep last 2 rows
    assert len(buf.table) == 2
    assert buf.table["x"].to_pylist() == [8, 9]


def test_retain_random():
    """Test that 'random' retention keeps a random subset of rows."""
    np.random.seed(0)  # deterministic test
    buf = DatasetBuffer()
    buf.table = make_table(10)
    buf._apply_retain_prior(0.5, "random")  # keep 5 rows
    assert len(buf.table) == 5
    # Check that values are a subset of original
    assert set(buf.table["x"].to_pylist()).issubset(set(range(10)))


def test_add_with_retain_prior():
    """Test appending with retention keeps prior data as expected."""
    np.random.seed(0)
    buffer = DatasetBuffer()
    buffer.table = make_table(10)
    buffer.append(
        [{"x": 100}, {"x": 101}],
        retain_prior=0.5,
        retain_prior_by="latest"
    )
    # retain latest 5 rows → [5,6,7,8,9]
    # then add 2 new rows → total 7
    assert len(buffer.table) == 7
    xs = buffer.table["x"].to_pylist()
    assert xs[:5] == [5, 6, 7, 8, 9]
    assert xs[-2:] == [100, 101]


def test_max_size_drops_oldest_rows():
    """testing dropping of oldest rows triggered by max size"""
    buffer = DatasetBuffer(max_size=3, drop_strategy="oldest")
    buffer.append([{"x": i} for i in range(10)])
    assert len(buffer) == 3
    assert buffer.table["x"].to_pylist() == [7, 8, 9]


def test_max_size_random_strategy_keeps_subset():
    """Testing of max size keeping subset"""
    np.random.seed(0)
    buffer = DatasetBuffer(max_size=4, drop_strategy="random")
    buffer.append([{"x": i} for i in range(10)])
    kept = buffer.table["x"].to_pylist()
    assert len(kept) == 4
    assert set(kept).issubset(set(range(10)))


def test_max_size_enforced_across_multiple_appends():
    """Testing max size is enforced"""
    buffer = DatasetBuffer(max_size=5, drop_strategy="oldest")
    buffer.append([{"x": i} for i in range(3)])
    buffer.append([{"x": 3}, {"x": 4}])
    assert len(buffer) == 5
    buffer.append([{"x": 5}, {"x": 6}])
    assert len(buffer) == 5
    assert buffer.table["x"].to_pylist() == [2, 3, 4, 5, 6]
