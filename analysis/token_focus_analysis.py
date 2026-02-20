import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config

from utils.load_hidden_states import load_hidden_states


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def select_layer(hidden_states, layer_index):
    """
    Select a layer from hidden states.
    """
    num_layers = len(hidden_states)

    if layer_index == -1:
        layer_index = num_layers - 1

    if layer_index >= num_layers or layer_index < 0:
        raise ValueError("Invalid layer index")

    return hidden_states[layer_index], layer_index


def extract_token_vectors(layer_state):
    """
    Remove batch dimension.
    """
    return layer_state[0]


def compute_token_norms(token_vectors):
    """
    Compute L2 norm per token.
    """
    return torch.norm(token_vectors, dim=1).numpy()


def compute_cosine_similarity_matrix(token_vectors):
    """
    Compute pairwise cosine similarity matrix.
    """
    seq_len = token_vectors.shape[0]
    matrix = torch.zeros((seq_len, seq_len))

    for i in range(seq_len):
        for j in range(seq_len):
            matrix[i, j] = F.cosine_similarity(
                token_vectors[i],
                token_vectors[j],
                dim=0
            )

    return matrix.numpy()


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_token_norms(tokens, norms, layer_index):
    plt.figure(figsize=(8, 4))
    plt.plot(tokens, norms)
    plt.title(f"Token Vector Norms (Layer {layer_index})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_cosine_matrix(tokens, cosine_matrix, layer_index):
    plt.figure(figsize=(6, 6))
    plt.imshow(cosine_matrix, aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Token Cosine Similarity (Layer {layer_index})")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_activation_heatmap(tokens, token_vectors, layer_index):
    activation_matrix = token_vectors.numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(activation_matrix, aspect="auto")
    plt.colorbar(label="Activation Value")
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Token Activation Heatmap (Layer {layer_index})")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.show()
    plt.close()


# --------------------------------------------------
# Main Analysis
# --------------------------------------------------

def run_token_focus_analysis():
    tokens, hidden_states = load_hidden_states()

    layer_state, layer_index = select_layer(
        hidden_states,
        config.LAYER_INDEX
    )

    token_vectors = extract_token_vectors(layer_state)

    norms = compute_token_norms(token_vectors)
    cosine_matrix = compute_cosine_similarity_matrix(token_vectors)


    plot_token_norms(tokens, norms, layer_index)
    plot_cosine_matrix(tokens, cosine_matrix, layer_index)
    plot_activation_heatmap(tokens, token_vectors, layer_index)

    print("Token focus analysis complete.")


# --------------------------------------------------

if __name__ == "__main__":
    run_token_focus_analysis()