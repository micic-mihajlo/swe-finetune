# Renderer System Reference

Tinker Cookbook renderer system for converting between messages and token sequences.

## Overview

Renderers handle bidirectional conversion:
- **Messages → Tokens**: For training and inference
- **Tokens → Messages**: For parsing model outputs

## Getting a Renderer

### Automatic Selection (Recommended)

```python
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
import tinker

model_name = "meta-llama/Llama-3.1-8B"
renderer_name = get_recommended_renderer_name(model_name)

tokenizer = tinker.get_tokenizer(model_name)
renderer = get_renderer(
    name=renderer_name,
    tokenizer=tokenizer,
    max_length=2048,
)
```

### Renderer Names

- `"chatml"`: ChatML format (many models)
- `"llama3"`: Llama 3 chat format
- `"qwen3"`: Qwen 3 chat format
- `"qwen3vl"`: Qwen 3 VL (vision-language)

## Core Methods

### build_supervised_example

Converts messages to training data with loss weights:

```python
from tinker_cookbook.renderers import TrainOnWhat

messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

example = renderer.build_supervised_example(
    messages=messages,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)

# Returns:
# - example.chunk: EncodedTextChunk with tokens
# - example.target_tokens: List[int]
# - example.weights: List[float] (0.0=ignore, 1.0=train)
```

### build_generation_prompt

For inference prompts:

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
]

prompt_chunk = renderer.build_generation_prompt(messages)
```

### get_stop_sequences

Returns stop tokens for generation:

```python
stop_sequences = renderer.get_stop_sequences()

sampling_params = SamplingParams(
    max_tokens=100,
    stop=stop_sequences,
)
```

### parse_response

Converts generated tokens back to message:

```python
message = renderer.parse_response(output_tokens)
# Returns: {"role": "assistant", "content": "..."}
```

## TrainOnWhat Enum

### ALL_ASSISTANT_MESSAGES

Trains on every assistant turn:

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},  # Train
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "Good!"},  # Train
]
```

**Use for**: General chat models, multi-turn dialogue

### LAST_ASSISTANT_MESSAGE

Trains only on final response:

```python
messages = [
    {"role": "user", "content": "Think step by step"},
    {"role": "assistant", "content": "Let me think..."},  # Skip
    {"role": "user", "content": "Continue"},
    {"role": "assistant", "content": "Answer: 4"},  # Train
]
```

**Use for**: Classification, reward modeling, chain-of-thought

## Message Format

### Text-Only

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"},
]
```

### Multi-Modal (Vision)

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_bytes},
            {"type": "text", "text": "What's in this image?"},
        ]
    },
    {"role": "assistant", "content": "A cat."}
]
```

## Using with Datasets

### With conversation_to_datum

```python
from tinker_cookbook.supervised.data import conversation_to_datum

datum = conversation_to_datum(
    messages=messages,
    renderer=renderer,
    max_length=2048,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
```

### With ChatDatasetBuilder

```python
@chz.chz
class MyDatasetBuilder(ChatDatasetBuilder):
    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self):
        # self.renderer automatically created from common_config
        def map_fn(row):
            return conversation_to_datum(
                messages=row["messages"],
                renderer=self.renderer,  # Access via self
                max_length=self.common_config.max_length,
                train_on_what=self.common_config.train_on_what,
            )
        # ...
```

## Common Patterns

### System Prompt Injection

```python
system_message = {"role": "system", "content": "You are an expert."}
messages = [system_message] + conversation_messages
```

### Few-Shot Examples

```python
few_shot = [
    {"role": "user", "content": "1+1?"},
    {"role": "assistant", "content": "2"},
]
actual = [
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "4"},
]
messages = few_shot + actual

# Only train on actual, not examples
example = renderer.build_supervised_example(
    messages=messages,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
)
```

## Best Practices

1. **Use automatic selection**: `get_recommended_renderer_name()`
2. **Don't construct formats manually**: Let renderer handle tokens
3. **Match renderer to model**: Wrong renderer = wrong format
4. **Use TrainOnWhat appropriately**: ALL for chat, LAST for tasks
5. **Use stop sequences**: Essential for proper generation
