# Tinker Low-Level API Reference

Manual training control using Tinker's ServiceClient and TrainingClient APIs.

## When to Use Low-Level API

- Custom training loop logic
- Fine-grained control over each training step
- Research experiments with non-standard training
- Online learning or RL with custom logic
- Direct access to tokenization and data conversion

## Core Setup

### ServiceClient and TrainingClient

```python
import tinker
from tinker import types

# Create service client
service_client = tinker.ServiceClient()

# Create LoRA training client
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B",
    rank=32,  # LoRA rank
    train_attn=True,
    train_mlp=True,
    train_unembed=False,
    seed=42,
)

# Get tokenizer
tokenizer = training_client.get_tokenizer()
```

## Data Preparation

### Creating Datum Objects

```python
from tinker.types import Datum, ModelInput, TensorData
import numpy as np

prompt_text = "Question: What is 2+2?\nAnswer: "
completion_text = "4"

prompt_tokens = tokenizer.encode(prompt_text)
completion_tokens = tokenizer.encode(completion_text)

all_tokens = prompt_tokens + completion_tokens
weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

# Target tokens shifted by 1 for next-token prediction
datum = types.Datum(
    model_input=types.ModelInput.from_ints(all_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": types.TensorData.from_numpy(
            np.array(all_tokens[1:], dtype=np.int64)
        ),
        "weights": types.TensorData.from_numpy(
            np.array(weights[1:], dtype=np.float32)
        ),
    }
)
```

**Key Points**:
- Target tokens are **shifted by 1** (model predicts next token)
- Weights are **shifted by 1** to match targets
- Use 0.0 weights for prompt, 1.0 for completion

## Training Loop

### Basic Training Pattern

```python
from tinker.types import AdamParams

adam_params = AdamParams(
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
)

for step in range(num_steps):
    batch = prepare_batch(data[step * batch_size:(step + 1) * batch_size])

    # Forward and backward pass
    fwdbwd_future = training_client.forward_backward(
        data=batch,
        loss_fn="cross_entropy",
    )

    # Optimizer step
    optim_future = training_client.optim_step(adam_params)

    # Wait for results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    print(f"Step {step}, Loss: {fwdbwd_result.loss}")

    if (step + 1) % save_every == 0:
        training_client.save_state(name=f"checkpoint-{step}")
```

### Loss Functions

**Supervised Learning**:
- `"cross_entropy"`: Standard next-token prediction loss

**Reinforcement Learning**:
- `"importance_sampling"`: Policy gradient with importance weighting
- `"ppo"`: Proximal Policy Optimization
- `"cispo"`: Clipped Importance Sampling Policy Optimization
- `"dro"`: Direct Reward Optimization

## State Management

### Saving and Loading

```python
# Save full training state (optimizer + weights)
training_client.save_state(name="checkpoint-1000")

# Load state to resume training
training_client.load_state(path="checkpoint-1000")

# Save weights for inference only
sampling_client = training_client.save_weights_and_get_sampling_client(
    name="my-model-final"
)
```

## Sampling and Evaluation

```python
from tinker.types import SamplingParams

prompt_input = types.ModelInput.from_ints(prompt_tokens)

sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    stop=["<|endoftext|>"],
    seed=42,
)

result = sampling_client.sample(
    prompt=prompt_input,
    sampling_params=sampling_params,
    num_samples=1,
)

output_text = tokenizer.decode(result.sequences[0].tokens)
```

## Complete Example

```python
import tinker
from tinker import types
import numpy as np
import chz

@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    data_file: str = "train.jsonl"
    num_steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-4
    lora_rank: int = 32
    save_every: int = 100

def prepare_batch(data_items, tokenizer):
    batch = []
    for item in data_items:
        prompt_tokens = tokenizer.encode(item["prompt"])
        completion_tokens = tokenizer.encode(item["completion"])
        all_tokens = prompt_tokens + completion_tokens
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(all_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": types.TensorData.from_numpy(
                    np.array(all_tokens[1:], dtype=np.int64)
                ),
                "weights": types.TensorData.from_numpy(
                    np.array(weights[1:], dtype=np.float32)
                ),
            }
        )
        batch.append(datum)
    return batch

def train(config: Config):
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    data = load_data(config.data_file)

    adam_params = types.AdamParams(learning_rate=config.learning_rate)

    for step in range(config.num_steps):
        batch_data = data[step * config.batch_size:(step + 1) * config.batch_size]
        batch = prepare_batch(batch_data, tokenizer)

        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        fwdbwd_result = fwdbwd_future.result()
        print(f"Step {step}, Loss: {fwdbwd_result.loss}")

        if (step + 1) % config.save_every == 0:
            training_client.save_state(name=f"checkpoint-{step+1}")

    training_client.save_weights_and_get_sampling_client(name="final-model")

if __name__ == "__main__":
    config = chz.nested_entrypoint(Config)
    train(config)
```

## Integration with Cookbook

Use Cookbook utilities with low-level training:

```python
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

tokenizer = tinker.get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = get_renderer(name="chatml", tokenizer=tokenizer, max_length=2048)

messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

# Use conversation_to_datum for easy conversion
datum = conversation_to_datum(
    messages=messages,
    renderer=renderer,
    max_length=2048,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
```
