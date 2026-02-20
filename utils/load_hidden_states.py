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

def load_all_hidden_states():
    """
    Load hidden states for all prompts in config.PROMPTS.
    """
    data = torch.load(Path(config.DATA_PATH))

    hidden_dict = {}

    for prompt in config.PROMPTS:
        if prompt not in data:
            raise ValueError(f"Prompt not found in data: {prompt}")

        entry = data[prompt]
        hidden_dict[prompt] = {
            "tokens": entry["tokens"],
            "hidden_states": entry["hidden_states"]
        }

    return hidden_dict
