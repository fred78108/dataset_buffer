"""Unit tests for the DatasetBuffer that are not data type specific.

python -m pytest -s test/test_buffer.py
"""

import numpy as np
import pyarrow as pa
import pytest
import torch
from dataset_buffer.buffer import DatasetBuffer


def make_table(n):
    return pa.Table.from_pylist([{"x": i} for i in range(n)])


def test_normalize_unsupported_type():
    buffer = DatasetBuffer()
    with pytest.raises(TypeError):
        buffer.append(object())


def test_add_dict_and_contains():
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    assert sample in buffer


def test_add_dict_and_not_contains():
    buffer = DatasetBuffer()
    sample = {"key": "value"}
    buffer.append(sample)
    invalid_sample = {"key": "not_there"}
    assert invalid_sample not in buffer


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


def test_add_diverse_types():
    buffer = DatasetBuffer()
    first_add = [{"key": i} for i in range(5)]
    second_add = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
    buffer.append(first_add)
    with pytest.raises(ValueError):
        buffer.append(second_add)


def test_retain_prior_zero():
    buf = DatasetBuffer()
    buf.table = make_table(10)

    buf._apply_retain_prior(0, "random")

    assert len(buf.table) == 0


def test_retain_prior_one():
    buf = DatasetBuffer()
    buf.table = make_table(10)

    buf._apply_retain_prior(1, "random")

    assert len(buf.table) == 10


def test_retain_oldest():
    buf = DatasetBuffer()
    buf.table = make_table(10)

    buf._apply_retain_prior(0.3, "oldest")  # keep first 3 rows

    assert len(buf.table) == 3
    assert buf.table["x"].to_pylist() == [0, 1, 2]


def test_retain_latest():
    buf = DatasetBuffer()
    buf.table = make_table(10)

    buf._apply_retain_prior(0.2, "latest")  # keep last 2 rows

    assert len(buf.table) == 2
    assert buf.table["x"].to_pylist() == [8, 9]


def test_retain_random():
    np.random.seed(0)  # deterministic test
    buf = DatasetBuffer()
    buf.table = make_table(10)

    buf._apply_retain_prior(0.5, "random")  # keep 5 rows

    assert len(buf.table) == 5
    # Check that values are a subset of original
    assert set(buf.table["x"].to_pylist()).issubset(set(range(10)))


def test_add_with_retain_prior():
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
