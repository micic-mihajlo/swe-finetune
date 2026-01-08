"""Tokenization utilities for Tinker training."""

from typing import Dict, Any, List
import numpy as np


def get_tokenizer_and_renderer(training_client, model_name: str, max_length: int = 4096):
    """Get tokenizer and renderer for a model.

    Args:
        training_client: Tinker TrainingClient to get tokenizer from
        model_name: Model name for renderer lookup
        max_length: Maximum sequence length
    """
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.model_info import get_recommended_renderer_name

    tokenizer = training_client.get_tokenizer()
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer, max_length=max_length)
    return tokenizer, renderer


def messages_to_datum(messages: List[Dict[str, str]], renderer, train_on_what: str = "all"):
    """Convert messages to Tinker Datum."""
    from tinker.types import Datum, ModelInput, TensorData
    from tinker_cookbook.renderers import TrainOnWhat

    train_on = TrainOnWhat.LAST_ASSISTANT_MESSAGE if train_on_what == "last" else TrainOnWhat.ALL_ASSISTANT_MESSAGES
    example = renderer.build_supervised_example(messages=messages, train_on_what=train_on)

    return Datum(
        model_input=ModelInput([example.chunk]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.array(example.target_tokens, dtype=np.int64)),
            "weights": TensorData.from_numpy(np.array(example.weights, dtype=np.float32)),
        },
    )


def batch_to_data(batch: List[Dict[str, Any]], renderer, train_on_what: str = "all") -> List:
    """Convert batch of examples to list of Datum objects."""
    data = []
    for example in batch:
        messages = example.get("messages", [])
        if messages:
            try:
                data.append(messages_to_datum(messages, renderer, train_on_what))
            except Exception as e:
                print(f"Warning: Skipping example: {e}")
    return data
