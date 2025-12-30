"""The public and primary interface into the DatasetBuffer"""

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Type
)
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
import torch
from PIL import Image
import io

# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

Normalized = Dict[str, Any]  # or your TypedDict union

NORMALIZERS: Dict[Type[Any], Callable[[Any], Normalized]] = {}


def register_normalizer(type_: Type[Any]):
    def decorator(fn: Callable[[Any], Normalized]):
        NORMALIZERS[type_] = fn
        return fn
    return decorator


@register_normalizer(Image.Image)
def normalize_pil(img: Image.Image) -> Normalized:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return {"image": buffer.getvalue()}


@register_normalizer(torch.Tensor)
def normalize_torch(t: torch.Tensor) -> Normalized:
    return {"tensor": t.cpu().numpy().tolist()}


# ----------------------------------------------------------------------
# The primary buffer
# ----------------------------------------------------------------------


class DatasetBuffer:
    """A dataset is dynamic and creates a "buffer" that allows for the
    modification of the dataset.
    """
    def __init__(self, data: Optional[Any] = None):
        self.table = pa.table({})  # empty table
        self._cur_type: Optional[Type] = None
        if data is not None:
            self.append(data)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict:
        return self.table.slice(idx, 1).to_pydict()

    def __iter__(self):
        for i in range(len(self.table)):
            yield self.table[i]

    def __contains__(self, item):
        """_summary_

        Args:
            item (_type_): _description_

        Returns:
            _type_: _description_
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
        return self.table.take(indices)

    def normalize(self, sample: Any) -> Normalized:
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

    def append(
        self,
        samples: Any,
        retain_prior: float = 1,
        retain_prior_by: Literal["random", "latest", "oldest"] = "random"
    ):
        # any prior adjustments needed?
        self._apply_retain_prior(retain_prior, retain_prior_by)
        # now setup
        if not isinstance(samples, list):
            samples = [samples]
        normalized = [self.normalize(s) for s in samples]
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

    def to_pydict(self) -> dict[str, list]:
        return self.table.to_pydict()

    def to_torch(self) -> dict[str, torch.Tensor]:
        batch = self.table.to_pydict()
        return {k: torch.tensor(v) for k, v in batch.items()}
