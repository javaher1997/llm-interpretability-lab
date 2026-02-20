import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

import config
# ---------------------------
# Configuration
# ---------------------------

MODEL_NAME = "gpt2"
OUTPUT_DIR = Path("data/hidden_states")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Load model & tokenizer
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(device)

model.eval()

# ---------------------------
# Extract hidden states
# ---------------------------

results = {}

with torch.no_grad():
    for prompt in config.PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # outputs.hidden_states is a tuple:
        # (embedding_layer, layer1, layer2, ..., layer12)
        hidden_states = outputs.hidden_states

        results[prompt] = {
            "input_ids": inputs["input_ids"].cpu(),
            "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            "hidden_states": [h.cpu() for h in hidden_states],
        }

        print(f"Processed prompt: '{prompt}'")
        print(f"  Number of layers: {len(hidden_states)}")
        print(f"  Hidden state shape (last layer): {hidden_states[-1].shape}")

# ---------------------------
# Save results
# ---------------------------

output_path = OUTPUT_DIR / "gpt2_hidden_states.pt"
torch.save(results, output_path)

print(f"\nHidden states saved to: {output_path.resolve()}")
