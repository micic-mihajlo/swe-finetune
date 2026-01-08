# Vision-Language Model Datasets Reference

Patterns for creating datasets for vision-language models (VLMs) in Tinker Cookbook.

## When to Use Vision Patterns

- Training vision-language models (Qwen3-VL, etc.)
- Working with image-text pairs
- Building classifiers or captioning models
- Fine-tuning on multi-modal data

## Vision Model Support

- **Qwen3-VL-235B** (MoE, most capable)
- **Qwen3-VL-30B** (MoE, cost-effective)
- **Qwen3-VL-8B** (Efficient for experimentation)

## Core Components

### Image Processing

```python
from tinker_cookbook.model_info import get_image_processor
import tinker

image_processor = get_image_processor("Qwen/Qwen2-VL-7B-Instruct")
```

### Vision Renderers

```python
from tinker_cookbook.renderers import get_renderer

renderer = get_renderer(
    name="qwen3vl",  # Vision-specific
    tokenizer=tokenizer,
    max_length=2048,
)
```

### ImageChunk in ModelInput

```python
from tinker.types import ModelInput, ImageChunk

with open("image.jpg", "rb") as f:
    image_bytes = f.read()

image_chunk = ImageChunk(image_bytes)
model_input = ModelInput([image_chunk, text_chunk])
```

## Custom Vision Dataset

```python
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker.types import Datum, ModelInput, TensorData
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.model_info import get_image_processor
import tinker
from PIL import Image
import io
import numpy as np

class VisionDataset(SupervisedDataset):
    def __init__(self, config):
        self.config = config
        self.tokenizer = tinker.get_tokenizer(config.model_name)
        self.renderer = get_renderer(
            name=config.renderer_name,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
        )
        self.image_processor = get_image_processor(config.model_name)
        self.data = self._load_data()

    def __iter__(self):
        for image_path, label in self.data:
            image_bytes = self._load_image(image_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_bytes},
                        {"type": "text", "text": "Classify this image."},
                    ]
                },
                {"role": "assistant", "content": label}
            ]

            example = self.renderer.build_supervised_example(
                messages=messages,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )

            yield Datum(
                model_input=ModelInput([example.chunk]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_numpy(
                        np.array(example.target_tokens, dtype=np.int64)
                    ),
                    "weights": TensorData.from_numpy(
                        np.array(example.weights, dtype=np.float32)
                    ),
                },
            )

    def _load_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()
```

## Image Loading Patterns

### From File System

```python
from PIL import Image
import io

def load_image_bytes(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
```

### From URL

```python
import requests

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
```

### From Numpy Array

```python
def numpy_to_image_bytes(array):
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
```

## Multi-Modal Message Formats

### Vision Classification

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_bytes},
            {"type": "text", "text": "Classify this image."},
        ]
    },
    {"role": "assistant", "content": "cat"}
]
```

### Image Captioning

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_bytes},
            {"type": "text", "text": "Describe this image."},
        ]
    },
    {"role": "assistant", "content": "A cat sitting on a couch."}
]
```

### Visual Question Answering

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_bytes},
            {"type": "text", "text": "How many cats are there?"},
        ]
    },
    {"role": "assistant", "content": "Two cats."}
]
```

## HuggingFace Vision Datasets

```python
from datasets import load_dataset
from io import BytesIO

dataset = load_dataset("cifar10", split="train")

def process_hf_image(example):
    image = example["img"]  # PIL Image
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

for item in dataset:
    image_bytes = process_hf_image(item)
    label = item["label"]
```

## Best Practices

1. **Use vision renderers**: `get_recommended_renderer_name(vlm_model)`
2. **Process images correctly**: Use `get_image_processor()`
3. **Convert to RGB**: Ensure consistent format
4. **Use LAST_ASSISTANT_MESSAGE**: For classification/captioning
5. **Start with small batches**: VLMs use more memory (4-8)
6. **Use MoE models**: Qwen3-VL for efficiency

## Common Imports

```python
from PIL import Image
import io
from tinker_cookbook.model_info import get_image_processor
from tinker.types import Datum, ModelInput, TensorData, ImageChunk
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
import tinker
import numpy as np
```
