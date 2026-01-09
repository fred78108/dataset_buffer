"""
DatasetBuffer: A dynamic, type-safe buffer for dataset management.

This module provides the primary public interface for the DatasetBuffer,
enabling efficient storage, normalization, and manipulation of datasets
using Apache Arrow tables. It supports appending, batching, and conversion
to various formats (e.g., PyTorch tensors), with extensible normalization
for custom data types.

Key features:
- Type-safe, schema-aligned buffer using Apache Arrow
- Extensible normalization for images, tensors, and custom types
- Retention strategies for managing buffer size and data freshness
- Easy conversion to Python dicts and PyTorch tensors

Intended for use in data pipelines, machine learning workflows, and
applications requiring dynamic, in-memory dataset management.

TODO:

1. revisit "store as string logic" need a way to handle mixed values.

"""

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Type,
    Union
)
import datasets
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
import torch
from PIL import Image
import io
from .structure import Stage
# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

Normalized = Union[Dict[str, Any], list[Any]]
NORMALIZERS: Dict[Type[Any], Callable[[Any], Normalized]] = {}


def register_normalizer(type_: Type[Any]):
    """
    Decorator to register a normalization function for a specific type.

    Args:
        type_ (Type[Any]): The type to register the normalizer for.

    Returns:
        Callable: The decorator that registers the function.
    """
    def decorator(fn: Callable[[Any], Normalized]):
        NORMALIZERS[type_] = fn
        return fn
    return decorator


@register_normalizer(Image.Image)
def normalize_pil(img: Image.Image) -> Normalized:
    """
    Normalizes a PIL Image by converting it to PNG bytes.

    Args:
        img (Image.Image): The PIL Image to normalize.

    Returns:
        Normalized: A dictionary with the image bytes.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return {"image": buffer.getvalue()}


@register_normalizer(torch.Tensor)
def normalize_torch(t: torch.Tensor) -> Normalized:
    """
    Normalizes a torch.Tensor by converting it to a list.

    Args:
        t (torch.Tensor): The tensor to normalize.

    Returns:
        Normalized: A dictionary with the tensor as a list.
    """
    return {"tensor": t.cpu().numpy().tolist()}


@register_normalizer(datasets.Dataset)
def normalize_hf_dataset(t: datasets.Dataset) -> Normalized:
    """
    Normalizes a HF datasets.Dataset by converting it to a list of dicts (row-wise).

    Args:
        t (datasets.Dataset): The Huggingface Dataset to normalize

    Returns:
        list[Normalized]: List of row dicts.
    """
    assert isinstance(t, datasets.Dataset)
    return t.to_list()

# ----------------------------------------------------------------------
# The primary buffer
# ----------------------------------------------------------------------


class DatasetBuffer:
    """
    A dynamic dataset buffer that allows for modification and management
    of datasets using Apache Arrow tables.

    Provides methods for appending, normalizing, batching, and converting
    data to various formats.
    """
    def __init__(
        self,
        data: Optional[Any] = None,
        max_size: Optional[int] = None,
        drop_strategy: Literal["random", "oldest"] = "oldest",
        store_as_string: bool = True
    ):
        """
        Initialize the DatasetBuffer.

        Args:
            data (Optional[Any], optional): Initial data to append.
                Defaults to None.
        """
        self.table = pa.table({})  # empty table
        self._append_stages: list[Stage] = []  # start with no appends
        self._prepare_stages: list[Stage] = []  # start with no prepares
        self.max_size = max_size
        self.drop_strategy: Literal["random", "oldest"] = drop_strategy
        self._cur_type: Optional[Type] = None
        self.store_as_string = store_as_string
        if data is not None:
            self.append(data)

    def __len__(self) -> int:
        """
        Returns the number of rows in the buffer.

        Returns:
            int: Number of rows.
        """
        return len(self.table)

    def target_item(self, idx: int) -> dict:
        return self.table.to_pylist()[idx]

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single row as a dictionary.

        Args:
            idx (int): Row index.

        Returns:
            dict: The row as a dictionary.
        """
        # return self.table.to_pylist()[idx]
        return self.table.slice(idx, 1).to_pydict()

    def __iter__(self):
        """
        Iterate over rows in the buffer.

        Yields:
            dict: Each row as a dictionary.
        """
        for row in self.table.to_pylist():
            yield row

    def __contains__(self, item):
        """
        Check if an item exists in the buffer.

        Args:
            item (dict): The item to check.

        Returns:
            bool: True if the item exists, False otherwise.
        """
        masks = [
            pc.equal(self.table[col], item[col])    # type: ignore
            for col in self.table.column_names
        ]
        combined = masks[0]
        for m in masks[1:]:
            combined = pc.call_function("and", (combined, m))
        return pc.call_function("any", combined).as_py()

    def get_batch(self, indices: list[int]) -> pa.Table:
        """
        Get a batch of rows by indices.

        Args:
            indices (list[int]): List of row indices.

        Returns:
            pa.Table: The batch as an Arrow table.
        """
        return self.table.take(indices)

    def set_append_stages(self, stages: list[Stage]) -> None:
        # safety check
        assert all([isinstance(stage, Stage) for stage in stages])
        self._append_stages = stages.copy()

    def set_prepare_stages(self, stages: list[Stage]) -> None:
        # safety check
        assert all([isinstance(stage, Stage) for stage in stages])
        self._prepare_stages = stages.copy()

    def normalize(self, sample: Any) -> Normalized:
        """
        Normalize a sample using registered normalizers or default strategies.

        Args:
            sample (Any): The sample to normalize.

        Returns:
            Normalized: The normalized sample.

        Raises:
            TypeError: If the sample type is unsupported.
        """
        for t, fn in NORMALIZERS.items():
            if isinstance(sample, t):
                return fn(sample)
        # fallback: dict-like or object-like
        if isinstance(sample, dict):
            return sample

        if hasattr(sample, "__dict__"):
            return vars(sample)
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    def _apply_retain_prior(self, retain_prior, retain_prior_by):
        """
        Retain a fraction of prior data in the buffer according to
        the strategy.

        Args:
            retain_prior (float): Fraction of data to retain (0 to 1).
            retain_prior_by (str): Strategy: 'latest', 'oldest', or 'random'.

        Raises:
            ValueError: If an unknown strategy is provided.
        """
        if retain_prior < 0:
            self.table = pa.table({})   # clear all
        if retain_prior >= 1 or len(self.table) == 0:
            return  # keep everything
        n = len(self.table)
        k = int(n * retain_prior)

        if k <= 0:
            self.table = self.table.slice(0, 0)  # empty table
            return

        if retain_prior_by == "latest":
            # Keep last k rows
            self.table = self.table.slice(n - k, k)

        elif retain_prior_by == "oldest":
            # Keep first k rows
            self.table = self.table.slice(0, k)

        elif retain_prior_by == "random":
            # Randomly choose k row indices
            idx = np.random.choice(n, size=k, replace=False)
            idx = np.sort(idx)  # Arrow requires sorted indices
            # Use take() for zero-copy row selection
            self.table = self.table.take(pa.array(idx))
        else:
            raise ValueError(f"Unknown retain_prior_by: {retain_prior_by}")

    def _reduce_size(self) -> None:
        n = len(self.table)
        if self.max_size is None:
            return  # no-op
        if n <= self.max_size:
            return  # no-op
        # we have exceeded the max side which is set
        reduce_by_count = self.max_size

        if self.drop_strategy == "oldest":
            # Keep the last `reduce_by_count` rows (drop the oldest)
            self.table = self.table.slice(n - reduce_by_count, reduce_by_count)

        elif self.drop_strategy == "random":
            idx = np.random.choice(n, size=reduce_by_count, replace=False)
            idx = np.sort(idx)
            self.table = self.table.take(pa.array(idx))

    def append(
        self,
        samples: Any,
        retain_prior: float = 1,
        retain_prior_by: Literal["random", "latest", "oldest"] = "random"
    ):
        """
        Append samples to the buffer, optionally retaining a fraction of
        prior data.

        Args:
            samples (Any): The sample(s) to append.
            retain_prior (float, optional): Fraction of prior data to retain.
                Defaults to 1.
            retain_prior_by (Literal["random", "latest", "oldest"], optional):
                Retention strategy. Defaults to "random".

        Raises:
            ValueError: If sample types are inconsistent.
        """
        # any prior adjustments needed?
        self._apply_retain_prior(retain_prior, retain_prior_by)
        # now setup
        if not isinstance(samples, list):
            samples = [samples]
        # call append stages
        for stage in self._append_stages:
            pass
        normalized = []
        for s in samples:
            norm = self.normalize(s)
            if isinstance(norm, list):
                normalized.extend(norm)
            else:
                normalized.append(norm)
        # print(normalized[0])
        if self._cur_type is None and len(samples) > 0:
            self._cur_type = type(samples[0])
        # safety check to ensure no mixing.
        if self._cur_type is not None:
            if not all(isinstance(row, self._cur_type) for row in samples):
                raise ValueError(
                    "All entries must match type %s after adding data",
                    self._cur_type)
        if not normalized:
            return
        all_keys = sorted({key for row in normalized for key in row})
        if self.store_as_string:
            aligned = [
                {key: str(row.get(key)) if row.get(key) is not None else None for key in all_keys}
                for row in normalized
            ]
            schema = pa.schema([(key, pa.string()) for key in all_keys])
            new_table = pa.Table.from_pylist(aligned, schema=schema)
        else:
            aligned = [
                {key: row.get(key) for key in all_keys} for row in normalized
            ]
            new_table = pa.Table.from_pylist(aligned)

        if len(self.table) == 0:
            self.table = new_table
        else:
            # Align schemas
            self.table = pa.concat_tables(
                [self.table, new_table],
                promote_options="default"
            )
        # reduce needed?
        if self.max_size is not None:
            self._reduce_size()

    def to_pydict(self) -> dict[str, list]:
        """
        Convert the buffer to a Python dictionary.

        Returns:
            dict[str, list]: The buffer as a dictionary of columns.
        """
        return self.table.to_pydict()

    def update(self, index: int, row: dict) -> None:
        """
        Updates a row at a specific index.

        This method is not highly performant for large tables as it
        involves slicing and concatenating immutable Arrow tables.

        Args:
            index (int): The index of the row to update.
            row (dict): The new data for the row.
        """
        if not (0 <= index < len(self.table)):
            raise IndexError("Index out of range")

        # Create a new table for the updated row
        updated_row_table = pa.Table.from_pylist([row])

        # Slice the original table
        table_before = self.table.slice(0, index)
        table_after = self.table.slice(index + 1)

        # Concatenate the parts
        tables_to_concat = []
        if len(table_before) > 0:
            tables_to_concat.append(table_before)

        tables_to_concat.append(updated_row_table)

        if len(table_after) > 0:
            tables_to_concat.append(table_after)

        if not tables_to_concat:
            self.table = pa.table({})
        else:
            self.table = pa.concat_tables(
                tables_to_concat, promote_options="default"
            )

    def to_torch(self) -> dict[str, torch.Tensor]:
        """
        Convert the buffer to a dictionary of torch.Tensors.

        Returns:
            dict[str, torch.Tensor]: The buffer as tensors.
        """
        batch = self.table.to_pydict()
        return {k: torch.tensor(v) for k, v in batch.items()}

    def to_hf_dataset(self) -> datasets.Dataset:
        """Converts the buffer to a Huggingface Dataset

        Returns:
            datasets.Dataset: _description_
        """
        return datasets.Dataset.from_dict(self.to_pydict())
