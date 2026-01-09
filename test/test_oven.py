"""Testing of the oven

python -m pytest -s test/test_oven.py
"""

import pytest
from unittest.mock import MagicMock, patch
from dataset_buffer.oven import _add_ingredient, _prepare_data, _process_step, bake
from dataset_buffer.buffer import DatasetBuffer
from dataset_buffer.structure import (
    DatasetRecipe,
    Ingredient,
    Instruction,
    PreparationStep,
)


@pytest.fixture
def sample_source():
    """Fixture for a sample data source."""
    return [{"id": i, "value": i * 10} for i in range(10)]


@pytest.fixture
def buffer():
    """Fixture for an empty list."""
    return []


def _mock_shuffle(x):
    """Mock shuffle function that does nothing."""
    return x


def test_add_ingredient_no_filter_no_quantity(buffer, sample_source):
    """Tests adding an ingredient without filtering or quantity limits."""
    ingredient = Ingredient(source=sample_source)
    _add_ingredient(ingredient, buffer)
    assert len(buffer) == len(sample_source)
    assert buffer == sample_source


def test_add_ingredient_with_quantity(buffer, sample_source):
    """Tests adding an ingredient with a specific quantity."""
    quantity = 5
    ingredient = Ingredient(source=sample_source, quantity=quantity)
    with patch("random.shuffle", side_effect=_mock_shuffle):
        _add_ingredient(ingredient, buffer)
    assert len(buffer) == quantity
    # After shuffle (mocked to do nothing), it should take the first 5
    assert [row["id"] for row in buffer] == [i for i in range(quantity)]


def test_add_ingredient_with_filter(buffer, sample_source):
    """Tests adding an ingredient with a filter function."""

    def filter_func(row):
        return row["id"] % 2 == 0

    ingredient = Ingredient(source=sample_source, filter_func=filter_func)
    _add_ingredient(ingredient, buffer)
    assert len(buffer) == 5
    assert all(row["id"] % 2 == 0 for row in buffer)


def test_add_ingredient_with_filter_and_quantity(buffer, sample_source):
    """Tests adding an ingredient with both a filter and a quantity."""

    def filter_func(row):
        return row["id"] > 4  # Should result in 5 items

    quantity = 3
    ingredient = Ingredient(
        source=sample_source, filter_func=filter_func, quantity=quantity
    )
    with patch("random.shuffle", side_effect=_mock_shuffle):
        _add_ingredient(ingredient, buffer)
    assert len(buffer) == quantity
    assert [row["id"] for row in buffer] == [5, 6, 7]


def test_prepare_data_raises_error_for_high_percentage(buffer):
    """Tests that _prepare_data raises a ValueError for percentage > 1."""

    def prep_func(row):
        return row

    prep_step = PreparationStep(percentage=1.1, prepare_func=prep_func)
    with pytest.raises(ValueError):
        _prepare_data(prep_step, buffer)


def test_prepare_data_with_percentage(sample_source):
    """Tests that _prepare_data applies a function to a percentage of
    the buffer."""
    buffer = list(sample_source)

    mock_prepare_func = MagicMock()
    percentage = 0.5
    prep_step = PreparationStep(
        percentage=percentage, prepare_func=mock_prepare_func
    )

    k = int(len(buffer) * percentage)
    # Mock random.sample to return predictable indices
    chosen_indices = list(range(k))
    with patch("random.sample", return_value=chosen_indices) as mock_sample:
        _prepare_data(prep_step, buffer)
        mock_sample.assert_called_once_with(range(len(buffer)), k=k)

    assert mock_prepare_func.call_count == k
    for i in chosen_indices:
        # Check that the mock was called with the rows corresponding to
        # the chosen indices
        mock_prepare_func.assert_any_call(sample_source[i])


def test_prepare_data_with_zero_percentage(sample_source):
    """Tests that _prepare_data does nothing when percentage is 0."""
    buffer = list(sample_source)

    mock_prepare_func = MagicMock()
    prep_step = PreparationStep(
        percentage=0, prepare_func=mock_prepare_func
    )

    with patch("random.sample") as mock_sample:
        _prepare_data(prep_step, buffer)
        mock_sample.assert_not_called()

    mock_prepare_func.assert_not_called()


def test_prepare_data_with_full_percentage(sample_source):
    """Tests that _prepare_data applies a function to the entire buffer."""
    buffer = list(sample_source)

    mock_prepare_func = MagicMock()
    prep_step = PreparationStep(
        percentage=1, prepare_func=mock_prepare_func
    )

    # Mock random.sample to return predictable indices
    with patch("random.sample", return_value=list(range(len(buffer)))) as mock_sample:
        _prepare_data(prep_step, buffer)
        mock_sample.assert_called_once_with(range(len(buffer)), k=len(buffer))

    assert mock_prepare_func.call_count == len(sample_source)
    # Verify it was called with the actual rows from the buffer
    for row in buffer:
        mock_prepare_func.assert_any_call(row)


@patch("dataset_buffer.oven._add_ingredient")
def test_process_step_add_ingredient(mock_add, buffer):
    """Tests that _process_step calls _add_ingredient for the correct instruction."""
    ingredient = Ingredient(source=[])
    instruction = Instruction(step="add_ingredient", details=ingredient)
    _process_step(instruction, buffer)
    mock_add.assert_called_once_with(data=ingredient, buffer=buffer)


@patch("dataset_buffer.oven._prepare_data")
def test_process_step_prepare_step(mock_prepare, buffer):
    """Tests that _process_step calls _prepare_data for the correct instruction."""

    def prep_func(r):
        return r

    prep_step = PreparationStep(percentage=0.1, prepare_func=prep_func)
    instruction = Instruction(step="prepare_step", details=prep_step)
    _process_step(instruction, buffer)
    mock_prepare.assert_called_once_with(data=prep_step, buffer=buffer)


def test_process_step_with_invalid_step(buffer):
    """Tests that _process_step handles an unknown step gracefully."""
    # Based on the current implementation, it should do nothing.
    instruction = Instruction(
        step="unknown_step", details=None)    # type: ignore
    try:
        _process_step(instruction, buffer)
    except Exception as e:
        pytest.fail(f"_process_step raised an exception for an unknown step: {e}")


@patch("dataset_buffer.oven._prepare_data")
def test_process_step_prepare_step_calls_prepare(mock_prepare, buffer):
    """Tests that _process_step calls _prepare_data for the correct instruction."""

    def prep_func(r):
        return r

    prep_step = PreparationStep(percentage=0.1, prepare_func=prep_func)
    instruction = Instruction(step="prepare_step", details=prep_step)
    _process_step(instruction, buffer)
    mock_prepare.assert_called_once_with(data=prep_step, buffer=buffer)


def test_bake_integration(sample_source):
    """An integration test for the bake function with multiple steps."""

    def filter_even(r):
        return r["id"] % 2 == 0

    def filter_odd(r):
        return r["id"] % 2 != 0

    # Step 1: Add all even-ID items
    ingredient1 = Ingredient(source=sample_source, filter_func=filter_even)
    instruction1 = Instruction(step="add_ingredient", details=ingredient1)

    # Step 2: Add 2 odd-ID items
    ingredient2 = Ingredient(
        source=sample_source, filter_func=filter_odd, quantity=2
    )
    instruction2 = Instruction(step="add_ingredient", details=ingredient2)

    # Step 3: Modify 50% of the buffer
    def tmp_prepare_func(row: dict) -> None:
        row["modified"] = True

    prep_step = PreparationStep(percentage=0.5, prepare_func=tmp_prepare_func)
    instruction3 = Instruction(step="prepare_step", details=prep_step)

    recipe = DatasetRecipe(
        instructions=[instruction1, instruction2, instruction3])

    # Mock shuffle to make test deterministic
    with patch("random.shuffle", side_effect=_mock_shuffle):
        final_buffer = bake(recipe)

    # Assertions
    # 5 evens + 2 odds = 7 total
    assert len(final_buffer) == 7

    # Check contents from ingredients
    ids = [row['id'] for row in final_buffer]
    assert set(ids) == {0, 2, 4, 6, 8, 1, 3}

    # Check preparation step
    modified_count = sum(1 for row in final_buffer if row.get("modified", False))
    assert modified_count == 3  # 50% of 7 is 3


def test_bake_with_empty_recipe():
    """Tests that bake function returns an empty list for an empty recipe."""
    recipe = DatasetRecipe(instructions=[])
    final_buffer = bake(recipe)
    assert isinstance(final_buffer, DatasetBuffer)
    assert len(final_buffer) == 0
