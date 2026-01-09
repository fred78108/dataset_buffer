# Dataset-Buffer (rename to Easy-bake dataset???)
> **Status:** 🚧 Under development

The dataset buffer is a small package for building a mutable buffer that can normalize heterogeneous samples and control how much historical data is retained between training epochs.

## Motivation
During my research I have need for adjusting a training dataset over multiple epochs. The dataset buffer addresses this challenge by providing an easy-to-use, predictable mechanism for accomplishing this dataset modification while still supporting consistent dataset loading.

## Key features
- **Automatic normalization** for `dict`, `torch.Tensor`, `PIL.Image.Image`, and `datasets.Dataset` samples, plus extensible custom normalizers.
- **Schema-safe appends** that reject mixed sample types once a buffer is initialized.
- **Retention policies** (`latest`, `oldest`, `random`) with fractional keep ratios to bound memory growth.
- **Arrow-native storage** enabling zero-copy batch operations and conversions to Python dicts, PyTorch tensors, or Hugging Face datasets.
- **Max size enforcement** with configurable drop strategies (`oldest`, `random`) to manage memory usage.

## Installation
The intention is to publish to PyPi once a stable enough code base. Until then, copy the repository and install via:

```bash
pip install -e .
```

## Quick start
```python
from dataset_buffer.buffer import DatasetBuffer
import torch
from PIL import Image

buffer = DatasetBuffer(max_size=100, drop_strategy="oldest")

# Append samples
buffer.append([
    {"label": 0, "meta": "first"},
    {"label": 1, "meta": "second"},
])
buffer.append(
    {"label": 1, "meta": "third"},
    retain_prior=0.5,
    retain_prior_by="latest"
)

# Access buffer information
print(len(buffer))          # Total rows retained
print(buffer[0])            # Arrow slice as dict
batch = buffer.get_batch([0, len(buffer) - 1])  # Get a batch of rows
torch_batch = buffer.to_torch()                # Convert to PyTorch tensors
```

## Normalization pipeline
1. The buffer searches registered normalizers via `@register_normalizer`.
2. Built-ins:
   - `torch.Tensor → {"tensor": np.ndarray}` (CPU, list-friendly).
   - `PIL.Image.Image → {"image": PNG bytes}`.
   - `datasets.Dataset → list[dict]` (row-wise conversion).
3. Dicts pass through unchanged; other objects fall back to `vars(obj)`.

Register custom normalizers:
```python
from dataset_buffer.buffer import register_normalizer

@register_normalizer(MySample)
def normalize_sample(sample: MySample):
    return {"text": sample.text, "score": sample.score}
```

## Retention strategies
| Parameter            | Description                                                |
|----------------------|------------------------------------------------------------|
| `retain_prior`       | Fraction of existing rows to keep (negative clears all).   |
| `retain_prior_by`    | `latest`, `oldest`, or `random` selection semantics.       |

Retention is applied **before** new samples are appended, ensuring memory limits are enforced per call.

## Max size enforcement
The `DatasetBuffer` supports a `max_size` parameter to limit the number of rows in the buffer. When the buffer exceeds this size, rows are dropped according to the `drop_strategy`:
- **`oldest`**: Drops the oldest rows.
- **`random`**: Drops a random subset of rows.

## Conversions and accessors
- `len(buffer)` / iteration yield Arrow row slices.
- `buffer[i]` returns a single-row dict.
- `buffer.get_batch(indices)` returns a `pyarrow.Table`.
- `buffer.to_pydict()` → column-wise Python lists.
- `buffer.to_torch()` → `torch.Tensor` per column (best for numeric data).
- `buffer.to_hf_dataset()` → Hugging Face `datasets.Dataset`.

## Example: Hugging Face Dataset Conversion
```python
from dataset_buffer.buffer import DatasetBuffer
from datasets import Dataset

buffer = DatasetBuffer()
hf_dataset = Dataset.from_dict({"col1": [1, 2], "col2": ["a", "b"]})
buffer.append(hf_dataset)

hf_converted = buffer.to_hf_dataset()
print(hf_converted)
```

## Baking datasets with recipes

Like baking a cake a dataset also needs a recipe. The `bake` function is a powerful utility for creating datasets from structured recipes. It processes a sequence of instructions, allowing you to define how data is added, transformed, and prepared in a controlled and repeatable manner.

### Key Concepts:
- **Recipes**: A `DatasetRecipe` object that contains a list of instructions for dataset creation.
- **Instructions**: Steps that define actions such as adding data (`add_ingredient`) or transforming data (`prepare_step`).
- **Ingredients**: Data sources with optional filters and quantity limits.
- **Preparation Steps**: Transformations applied to a subset of the dataset.

### Workflow:
1. Define a `DatasetRecipe` with a sequence of instructions.
2. Use the `bake` function to process the recipe and generate a `DatasetBuffer`.
3. Access and manipulate the resulting dataset as needed.

### Benefits:
- Modular and reusable dataset creation.
- Fine-grained control over data filtering, sampling, and transformation.
- Seamless integration with the `DatasetBuffer` for further operations.

Refer to the codebase documentation for detailed examples and usage patterns.
