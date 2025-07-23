import matplotlib.pyplot as plt
import numpy as np
import sys

from load import load


def plot_final_field_values(metadata, data):
    data = np.array(data)
    positionx = np.arange(data.shape[0])
    positiony = np.arange(data.shape[1])
    X, Y = np.meshgrid(positionx, positiony)
    spinX = np.cos(data * 2 * np.pi)
    spinY = np.sin(data * 2 * np.pi)
    X = X - 0.5 * spinX
    Y = Y - 0.5 * spinY

    plt.gca().set_aspect("equal", adjustable="box")
    plt.quiver(
        X,
        Y,
        spinX,
        spinY,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
    )
    plt.title(f"XY Spin Model\n{metadata}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks([])
    plt.yticks([])
    metadata_str = metadata.replace(", ", "_").replace("=", "_")
    plt.savefig(f"figs/xy_spin_field_{metadata_str}.png", dpi=300)
    plt.show()


def plot_values_by_step(metadata, data):
    data = np.array(data)
    steps = data[:, 0]
    energy = data[:, 1]
    magnetization = data[:, 2]
    positive_vortices = data[:, 3]
    negative_vortices = data[:, 4]

    if np.any(positive_vortices != negative_vortices):
        print("Warning: Positive and negative vortices are not equal.")
        diff = positive_vortices != negative_vortices
        steps_diff = steps[diff]
        positive_diff = positive_vortices[diff]
        negative_diff = negative_vortices[diff]
        for step, pos, neg in zip(steps_diff, positive_diff, negative_diff):
            print(f"Step {step}: Positive Vortices = {pos}, Negative Vortices = {neg}")

    plt.figure(figsize=(10, 8), constrained_layout=True)
    plt.suptitle(f"XY Spin Model Metrics by Step\n{metadata}")

    plt.subplot(3, 1, 1)
    plt.plot(steps, energy, label="Energy", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy by Step")

    plt.subplot(3, 1, 2)
    plt.plot(steps, magnetization, label="Magnetization", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Magnetization")
    plt.title("Magnetization by Step")

    plt.subplot(3, 1, 3)
    plt.plot(
        steps,
        positive_vortices,
        label="Vortices",
        color="blue",
        marker=".",
        linestyle="None",
    )
    plt.xlabel("Step")
    plt.ylabel("Vortices")
    plt.title("Vortices by Step")

    plt.tight_layout()
    metadata_str = metadata.replace(", ", "_").replace("=", "_")
    plt.savefig(f"figs/xy_spin_metrics_{metadata_str}.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python plot_xyspin.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    metadata, data = load(filename)
    print(metadata)
    for key, value in data.items():
        if key.endswith("_metadata"):
            print(key)
            print(value)
    plot_values_by_step(metadata, data["metropolis_sampling"])
    plot_final_field_values(metadata, data["final_field_values"])
