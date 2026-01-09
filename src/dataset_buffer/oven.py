"""Used to "bake" a dataset recipe"""

import random
from typing import List, Literal, Union
from .structure import (
    DatasetRecipe,
    Ingredient,
    Instruction,
    PreparationStep
)
from .buffer import DatasetBuffer


def _add_ingredient(data: Ingredient, buffer: List) -> None:
    if data.filter_func:
        to_add = [row for row in data.source if data.filter_func(row)]
    else:
        to_add = list(data.source)

    if data.quantity:
        random.shuffle(to_add)
        to_add = to_add[0:data.quantity]
    buffer.extend(to_add)


def _prepare_data(data: PreparationStep, buffer: List) -> None:
    if data.percentage > 1:
        raise ValueError("Maximum percentage is 1 for preperation step")
    if data.percentage > 0:
        k = int(len(buffer) * data.percentage)
        # Get indices to update instead of copies of rows
        indices_to_update = random.sample(range(len(buffer)), k=k)
        for i in indices_to_update:
            # Get the row, modify it, and then update it in the buffer
            row = buffer[i]
            data.prepare_func(row)
            buffer[i] = row


def _process_step(instruction: Instruction, buffer: List) -> None:
    if instruction.step == "add_ingredient":
        assert isinstance(instruction.details, Ingredient)
        _add_ingredient(data=instruction.details, buffer=buffer)
    elif instruction.step == "prepare_step":
        assert isinstance(instruction.details, PreparationStep)
        _prepare_data(data=instruction.details, buffer=buffer)


def bake(
    recipe: DatasetRecipe,
    return_as: Literal["list", "DatasetBuffer"] = "DatasetBuffer"
) -> Union[DatasetBuffer, list]:
    """Bakes a data recipe into a list.

    Args:
        recipe (DatasetRecipe): The recipe to process

    Returns:
        List: The processed dataset
    """
    buffer: List = []    # empty buffer
    for bake_step in recipe.instructions:
        _process_step(instruction=bake_step, buffer=buffer)
    if return_as == "DatasetBuffer":
        return DatasetBuffer(data=buffer)
    return buffer
