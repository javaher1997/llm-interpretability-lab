import torch
import torch.nn.functional as F
from pathlib import Path

# ---------------------------
# Configuration
# ---------------------------

DATA_PATH = Path("data/hidden_states/gpt2_hidden_states.pt")

PROMPT = "The capital of France is"
TARGET_TOKEN = "ĠFrance"

# ---------------------------
# Load hidden states
# ---------------------------

data = torch.load(DATA_PATH)

if PROMPT not in data:
    raise ValueError(f"Prompt not found in data: {PROMPT}")

entry = data[PROMPT]

tokens = entry["tokens"]
hidden_states = entry["hidden_states"]

num_layers = len(hidden_states)

print(f"Loaded hidden states for prompt: '{PROMPT}'")
print(f"Number of layers: {num_layers}")
print(f"Tokens: {tokens}")

# ---------------------------
# Locate target token
# ---------------------------

if TARGET_TOKEN not in tokens:
    raise ValueError(f"Target token '{TARGET_TOKEN}' not found in tokens")

token_index = tokens.index(TARGET_TOKEN)
print(f"Target token '{TARGET_TOKEN}' found at position {token_index}")

# ---------------------------
# Extract layer-wise vectors
# ---------------------------

layer_vectors = []

for layer_idx, layer_state in enumerate(hidden_states):
    # shape: [batch, seq_len, hidden_dim]
    token_vector = layer_state[0, token_index]  # remove batch dim
    layer_vectors.append(token_vector)

# ---------------------------
# Compute norms and similarities
# ---------------------------

print("\nLayer-wise token representation analysis:\n")

for i in range(num_layers):
    vec = layer_vectors[i]
    norm = torch.norm(vec).item()

    if i == 0:
        print(f"Layer {i:02d} | norm = {norm:.4f}")
    else:
        prev_vec = layer_vectors[i - 1]
        cosine_sim = F.cosine_similarity(vec, prev_vec, dim=0).item()
        print(
            f"Layer {i:02d} | norm = {norm:.4f} | "
            f"cosine similarity to layer {i-1:02d} = {cosine_sim:.4f}"
        )

# ---------------------------
# Interpretation note
# ---------------------------

print(
    "\nInterpretation guide:\n"
    "- Increasing norms suggest feature enrichment\n"
    "- High cosine similarity between later layers suggests semantic stabilization\n"
    "- Large early changes indicate structural/syntactic processing\n"
)
