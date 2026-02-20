import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

import config

from utils.load_hidden_states import load_hidden_states
# ---------------------------
# Global Matplotlib Styling
# ---------------------------

plt.style.use("default")
plt.rcParams["axes.edgecolor"] = config.COLOR_SECONDARY
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = config.FONT_SIZE_LABEL

# ---------------------------
# Token Extraction
# ---------------------------

def extract_token_vectors(tokens, hidden_states):
    if config.TARGET_TOKEN not in tokens:
        raise ValueError(
            f"Target token '{config.TARGET_TOKEN}' not found in tokens"
        )

    token_index = tokens.index(config.TARGET_TOKEN)

    layer_vectors = [
        layer_state[0, token_index]
        for layer_state in hidden_states
    ]

    return token_index, layer_vectors


# ---------------------------
# Metrics Computation
# ---------------------------

def compute_metrics(layer_vectors):
    norms = []
    cosine_sims = []

    for i, vec in enumerate(layer_vectors):
        norms.append(torch.norm(vec).item())

        if i == 0:
            cosine_sims.append(1.0)
        else:
            prev_vec = layer_vectors[i - 1]
            cosine_sims.append(
                F.cosine_similarity(vec, prev_vec, dim=0).item()
            )

    return norms, cosine_sims


# ---------------------------
# Visualization Functions
# ---------------------------

def plot_norms(norms):
    layers = list(range(len(norms)))

    plt.figure(figsize=config.FIG_SIZE_SMALL, dpi=config.DPI)

    plt.plot(
        layers,
        norms,
        color=config.COLOR_PRIMARY,
        linewidth=config.LINE_WIDTH,
        marker="o",
        markersize=config.MARKER_SIZE
    )

    plt.title(
        f"Vector Norm Across Layers\nToken: '{config.TARGET_TOKEN}'",
        fontsize=config.FONT_SIZE_TITLE,
        color=config.COLOR_ACCENT
    )

    plt.xlabel("Layer")
    plt.ylabel("Vector Norm")

    plt.xticks(color=config.COLOR_SECONDARY)
    plt.yticks(color=config.COLOR_SECONDARY)

    plt.tight_layout()
    plt.show()


def plot_cosine_similarity(cosine_sims):
    layers = list(range(1, len(cosine_sims)))

    plt.figure(figsize=config.FIG_SIZE_SMALL, dpi=config.DPI)

    plt.plot(
        layers,
        cosine_sims[1:],
        color=config.COLOR_SECONDARY,
        linewidth=config.LINE_WIDTH,
        marker="o",
        markersize=config.MARKER_SIZE
    )

    plt.title(
        f"Layer-to-Layer Cosine Similarity\nToken: '{config.TARGET_TOKEN}'",
        fontsize=config.FONT_SIZE_TITLE,
        color=config.COLOR_ACCENT
    )

    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1.05)

    plt.xticks(color=config.COLOR_SECONDARY)
    plt.yticks(color=config.COLOR_SECONDARY)

    plt.tight_layout()
    plt.show()


def plot_heatmap(layer_vectors):
    activation_matrix = torch.stack(layer_vectors).numpy()

    plt.figure(figsize=config.FIG_SIZE_LARGE, dpi=config.DPI)

    plt.imshow(
        activation_matrix,
        aspect="auto",
        cmap="Blues"
    )

    cbar = plt.colorbar()
    cbar.set_label("Activation Value")

    plt.title(
        f"Neuron Activations Across Layers\nToken: '{config.TARGET_TOKEN}'",
        fontsize=config.FONT_SIZE_TITLE,
        color=config.COLOR_ACCENT
    )

    plt.xlabel("Hidden Dimension")
    plt.ylabel("Layer")

    plt.xticks(color=config.COLOR_SECONDARY)
    plt.yticks(color=config.COLOR_SECONDARY)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Main Execution
# ---------------------------

def main():
    print(f"\nLoaded hidden states for prompt: '{config.PROMPT}'")

    tokens, hidden_states = load_hidden_states()
    print(f"Number of layers: {len(hidden_states)}")
    print(f"Tokens: {tokens}")

    token_index, layer_vectors = extract_token_vectors(
        tokens,
        hidden_states
    )

    print(
        f"Target token '{config.TARGET_TOKEN}' found at position {token_index}"
    )

    norms, cosine_sims = compute_metrics(layer_vectors)

    print("\nLayer-wise token representation analysis:\n")
    for i in range(len(norms)):
        print(
            f"Layer {i:02d} | norm = {norms[i]:.4f} | "
            f"cosine similarity = {cosine_sims[i]:.4f}"
        )

    plot_norms(norms)
    plot_cosine_similarity(cosine_sims)
    plot_heatmap(layer_vectors)

    print(
        "\nInterpretation guide:\n"
        "- Rising norms indicate feature enrichment\n"
        "- Cosine similarity approaching 1 indicates semantic stabilization\n"
        "- Early low similarity suggests syntactic processing\n"
        "- Heatmap reveals neuron specialization patterns\n"
    )


if __name__ == "__main__":
    main()