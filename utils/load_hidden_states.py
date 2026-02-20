import torch
from pathlib import Path
import config


def load_hidden_states():
    """
    Load tokens and hidden states for the configured prompt.
    """
    data_path = Path(config.DATA_PATH)
    data = torch.load(data_path)

    if config.PROMPT not in data:
        raise ValueError(f"Prompt not found in data: {config.PROMPT}")

    entry = data[config.PROMPT]
    return entry["tokens"], entry["hidden_states"]