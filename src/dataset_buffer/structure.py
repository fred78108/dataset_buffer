"""Shared structures"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Iterable, Optional, Union

Action = Literal["add_ingredient", "prepare_step"]
DataSource = Literal["in-memory", "on-disk"]

# TODO: do no I need to restrict data types here?


@dataclass
class Ingredient:
    """Add some data """
    source: Iterable
    filter_func: Optional[Callable[[Any], bool]] = None
    quantity: Optional[int] = None


@dataclass
class PreparationStep:
    percentage: float
    prepare_func: Callable[[Any], None]


@dataclass
class Instruction:
    step: Action
    details: Union[PreparationStep, Ingredient]


@dataclass
class DatasetRecipe:
    instructions: list[Instruction]
    description: Optional[str] = None


@dataclass
class Stage:
    """An entry for staging of data. Allows extending functionality and
    avoiding duplication of processing steps such as on-append."""
    name: str
    func: Callable[[Iterable], None]
    enabled: bool = True
