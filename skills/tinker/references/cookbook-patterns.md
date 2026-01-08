# Tinker Cookbook Patterns Reference

This document covers all high-level Tinker Cookbook patterns for supervised fine-tuning.

## Configuration Patterns

### Pattern 1: @chz.chz Decorator (Class-Based)

Use for straightforward CLI configuration:

```python
import chz
import asyncio
from tinker_cookbook.supervised import train

@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    file_path: str = "data.jsonl"
    max_length: int = 2048

async def train_async(cli_config: CLIConfig):
    config = train.Config(
        model_name=cli_config.model_name,
        log_path="/tmp/training",
        dataset_builder=build_dataset(cli_config),
    )
    await train.main(config)

def main():
    cli_config = chz.nested_entrypoint(CLIConfig)
    asyncio.run(train_async(cli_config))

if __name__ == "__main__":
    main()
```

### Pattern 2: Blueprint (Function-Based)

Use for template-based configuration with overrides:

```python
import chz
import sys
import asyncio
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.model_info import get_recommended_renderer_name

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = get_recommended_renderer_name(model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = MyDatasetBuilder(common_config=common_config)

    return chz.Blueprint(train.Config).apply({
        "log_path": "/tmp/training",
        "model_name": model_name,
        "dataset_builder": dataset_builder,
        "learning_rate": 2e-4,
        "lr_schedule": "cosine",
        "num_epochs": 3,
        "lora_rank": 32,
    })

if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    asyncio.run(train.main(blueprint.make()))
```

## Dataset Builder Patterns

### Pattern 3: HuggingFace Dataset Builder

```python
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.renderers import TrainOnWhat
import datasets
import chz

@chz.chz
class MyDatasetBuilder(ChatDatasetBuilder):
    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self):
        hf_dataset = datasets.load_dataset("HuggingFaceH4/no_robots", split="train")
        split = hf_dataset.train_test_split(test_size=0.1, seed=42)

        def map_fn(row):
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ]
            return conversation_to_datum(
                messages=messages,
                renderer=self.renderer,
                max_length=self.common_config.max_length,
                train_on_what=self.common_config.train_on_what,
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            hf_dataset=split["train"],
            batch_size=self.common_config.batch_size,
            map_fn=map_fn,
        )
        test_dataset = SupervisedDatasetFromHFDataset(
            hf_dataset=split["test"],
            batch_size=self.common_config.batch_size,
            map_fn=map_fn,
        )
        return train_dataset, test_dataset
```

### Pattern 4: File-Based Dataset Loading

```python
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import os

def build_dataset(cli_config):
    if not os.path.exists(cli_config.file_path):
        raise FileNotFoundError(f"Data file not found: {cli_config.file_path}")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=get_recommended_renderer_name(cli_config.model_name),
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    return FromConversationFileBuilder(
        common_config=common_config,
        file_path=cli_config.file_path,
    )
```

**JSONL File Format**:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Pattern 5: Streaming Datasets

For large datasets that don't fit in memory:

```python
from tinker_cookbook.supervised.data import StreamingSupervisedDatasetFromHFDataset
import datasets

@chz.chz
class StreamingDatasetBuilder(ChatDatasetBuilder):
    common_config: ChatDatasetBuilderCommonConfig
    max_prompts: int = 100000
    buffer_size: int = 10000

    def __call__(self):
        ds = datasets.load_dataset(
            "open-thoughts/OpenThoughts3-1.2M",
            split="train",
            streaming=True  # Important!
        )

        def map_fn(row):
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["response"]},
            ]
            return conversation_to_datum(
                messages=messages,
                renderer=self.renderer,
                max_length=self.common_config.max_length,
                train_on_what=self.common_config.train_on_what,
            )

        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            hf_dataset=ds,
            batch_size=self.common_config.batch_size,
            length=self.max_prompts,  # Required for streaming
            map_fn=map_fn,
            buffer_size=self.buffer_size,
        )
        return train_dataset, test_dataset
```

### Pattern 6: Custom Dataset Implementation

```python
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker.types import Datum, ModelInput, TensorData
from tinker_cookbook.renderers import get_renderer
import tinker
import numpy as np

class CustomDataset(SupervisedDataset):
    def __init__(self, config):
        self.config = config
        self.tokenizer = tinker.get_tokenizer(config.model_name)
        self.renderer = get_renderer(
            name=config.renderer_name,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
        )
        self.data = self._load_data()

    def __len__(self):
        return len(self.data) // self.config.batch_size

    def __iter__(self):
        for item in self.data:
            messages = self._preprocess_item(item)
            example = self.renderer.build_supervised_example(
                messages=messages,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
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
```

## Training Configuration

### train.Config Fields

Required:
- `model_name`: Base model identifier
- `log_path`: Directory for logs and checkpoints
- `dataset_builder`: Instance of dataset builder

Common hyperparameters:
- `learning_rate`: Default 2e-4 for LoRA
- `lr_schedule`: "cosine", "linear", or "constant"
- `num_epochs`: Number of training epochs
- `lora_rank`: LoRA rank (default: 32)
- `save_every`: Save checkpoint every N steps
- `eval_every`: Run evaluation every N steps

### ChatDatasetBuilderCommonConfig Fields

- `model_name_for_tokenizer`: Model name for loading tokenizer
- `renderer_name`: Chat format
- `max_length`: Maximum sequence length in tokens
- `batch_size`: Batch size for dataset
- `train_on_what`: `TrainOnWhat` enum value

### TrainOnWhat Enum

- `TrainOnWhat.ALL_ASSISTANT_MESSAGES`: Train on all assistant turns
- `TrainOnWhat.LAST_ASSISTANT_MESSAGE`: Train only on final response

## Key Imports Reference

```python
import chz
import asyncio
import datasets

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    StreamingSupervisedDatasetFromHFDataset,
    FromConversationFileBuilder,
    conversation_to_datum,
)
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker.types import Datum, ModelInput, TensorData
import tinker
```
