import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
from utils.load_hidden_states import load_all_hidden_states

# --------------------------------------------------
# Utilities
# --------------------------------------------------


def get_final_token_vectors(hidden_states):
    """
    Extract final token vector for each layer.
    """
    final_vectors = []

    for layer in hidden_states:
        # shape: (1, seq_len, hidden_dim)
        layer_state = layer[0]  # remove batch dimension
        final_token_vector = layer_state[-1]  # last token
        final_vectors.append(final_token_vector)

    return final_vectors


def compute_layerwise_distances(vectors_a, vectors_b):
    """
    Compute L2 distance between two prompts across layers.
    """
    distances = []

    for va, vb in zip(vectors_a, vectors_b):
        distance = torch.norm(va - vb).item()
        distances.append(distance)

    return distances


def compute_layerwise_cosine(vectors_a, vectors_b):
    """
    Compute cosine similarity between two prompts across layers.
    """
    cosines = []

    for va, vb in zip(vectors_a, vectors_b):
        cosine = F.cosine_similarity(va, vb, dim=0).item()
        cosines.append(cosine)

    return cosines


def main():
    hidden_dict = load_all_hidden_states()

    prompts = config.PROMPTS

    if len(prompts) < 2:
        raise ValueError("Need at least 2 prompts for cross-prompt analysis")

    # Generate all pairwise combinations
    for prompt_a, prompt_b in itertools.combinations(prompts, 2):

        vectors_a = get_final_token_vectors(hidden_dict[prompt_a]["hidden_states"])
        vectors_b = get_final_token_vectors(hidden_dict[prompt_b]["hidden_states"])

        distances = compute_layerwise_distances(vectors_a, vectors_b)
        cosines = compute_layerwise_cosine(vectors_a, vectors_b)

        # Distance plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(distances)), distances)
        plt.xlabel("Layer")
        plt.ylabel("L2 Distance")
        plt.title(f"Layerwise Distance\n'{prompt_a}' vs '{prompt_b}'")
        plt.tight_layout()
        plt.show()
        plt.close()

        # Cosine plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(cosines)), cosines)
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Layerwise Cosine Similarity\n'{prompt_a}' vs '{prompt_b}'")
        plt.tight_layout()
        plt.show()
        plt.close()

    print("Cross-prompt analysis complete.")

if __name__ == "__main__":
    main()