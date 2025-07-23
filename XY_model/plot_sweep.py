import matplotlib.pyplot as plt
import numpy as np
import sys

from load import load


def plot_values_by_beta(metadata, data):
    """
    Plot energy, magnetization, and vortices as a function of beta (inverse temperature).

    Expected data format:
    Each row: [beta, energy, magnetization, positive_vortices, negative_vortices]
    """
    data = np.array(data)
    beta = data[:, 0]
    energy = data[:, 1]
    magnetization = data[:, 2]
    positive_vortices = data[:, 3]
    negative_vortices = data[:, 4]

    if np.any(positive_vortices != negative_vortices):
        print("Warning: Positive and negative vortices are not equal.")
        diff = positive_vortices != negative_vortices
        beta_diff = beta[diff]
        positive_diff = positive_vortices[diff]
        negative_diff = negative_vortices[diff]
        for b, pos, neg in zip(beta_diff, positive_diff, negative_diff):
            print(f"Beta {b:.2f}: Positive Vortices = {pos}, Negative Vortices = {neg}")

    plt.figure(figsize=(6, 6), constrained_layout=True)
    plt.suptitle(f"XY Spin Model Beta Sweep\n{metadata}")

    # Energy vs Beta
    plt.subplot(3, 1, 1)
    plt.plot(beta, energy, "o", label="Energy", color="blue", markersize=4)
    plt.xlabel("β (inverse temperature)")
    plt.ylabel("Energy")
    plt.title("Energy vs β")
    plt.grid(True, alpha=0.3)

    # Magnetization vs Beta
    plt.subplot(3, 1, 2)
    plt.plot(beta, magnetization, "o", label="Magnetization", color="red", markersize=4)
    plt.xlabel("β (inverse temperature)")
    plt.ylabel("Magnetization")
    plt.title("Magnetization vs β")
    plt.grid(True, alpha=0.3)

    # Vortices vs Beta
    plt.subplot(3, 1, 3)
    plt.plot(
        beta, positive_vortices, "o", label="Vortices", color="green", markersize=4
    )
    plt.xlabel("β (inverse temperature)")
    plt.ylabel("Number of Vortices")
    plt.title("Vortices vs β")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    metadata_str = metadata.replace(", ", "_").replace("=", "_")
    plt.savefig(f"figs/xy_spin_beta_sweep_{metadata_str}.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python plot_sqeep.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    metadata, data = load(filename)
    print(metadata)
    for key, value in data.items():
        if key.endswith("_metadata"):
            print(key)
            print(value)

    plot_values_by_beta(metadata, data["beta_sweep_results"])
