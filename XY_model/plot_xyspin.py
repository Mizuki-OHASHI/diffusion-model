import matplotlib.pyplot as plt
import numpy as np
import sys

from load import load


def plot_final_field_values(metadata, data):
    data = np.array(data)
    positionx = np.arange(data.shape[0])
    positiony = np.arange(data.shape[1])
    X_, Y_ = np.meshgrid(positionx, positiony)
    spinX = np.cos(data * 2 * np.pi)
    spinY = np.sin(data * 2 * np.pi)
    X = X_ - 0.5 * spinX
    Y = Y_ - 0.5 * spinY

    positive, negative = find_vortex_positions_vectorized(
        data, center_of_plaquette=True
    )

    xstart, xwnd = 3, 3
    ystart, ywnd = 1, 2

    plt.gca().set_aspect("equal", adjustable="box")
    plt.quiver(X, Y, spinX, spinY, angles="xy", scale_units="xy", scale=1, color="blue")
    # plt.scatter(X_, Y_, color="blue", label="Positive Vortices", marker=".", s=10)
    plt.scatter(
        [x for _, x in positive],
        [y for y, _ in positive],
        color="red",
        label="Positive Vortices",
        marker="+",
        s=84,
    )
    plt.scatter(
        [x for _, x in negative],
        [y for y, _ in negative],
        color="red",
        label="Negative Vortices",
        marker="x",
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


def find_vortex_positions_vectorized(
    field: np.ndarray, center_of_plaquette: bool = False
) -> tuple[list, list]:
    """
    Finds vortex positions in a 2D scalar field using a vectorized algorithm.

    Args:
        field (np.ndarray): A 2D numpy array of shape (nx, ny) representing the
                            angle of spins at each point (in units of rad/2π).
        center_of_plaquette (bool): If True, returns the center of the plaquette
                                    (i + 0.5, j + 0.5) as a float tuple.
                                    If False (default), returns the bottom-left
                                    integer coordinate (i, j).

    Returns:
        tuple[list, list]: A tuple containing two lists:
                           - A list of positive vortex positions.
                           - A list of negative vortex positions.
    """
    # 1. 各プラケットの四隅の角度を取得
    # np.roll を使い、周期境界条件を考慮して配列をずらす
    theta0 = field  # (i, j)
    theta1 = np.roll(field, -1, axis=0)  # (i+1, j)
    theta2 = np.roll(theta1, -1, axis=1)  # (i+1, j+1)
    theta3 = np.roll(field, -1, axis=1)  # (i, j+1)

    # 2. プラケットを反時計回りに一周したときの各位相差を計算
    d_theta = np.zeros(field.shape + (4,))
    d_theta[..., 0] = theta1 - theta0
    d_theta[..., 1] = theta2 - theta1
    d_theta[..., 2] = theta3 - theta2
    d_theta[..., 3] = theta0 - theta3  # 最後の辺 (i, j+1) -> (i, j)

    # 3. 位相差を [-0.5, 0.5] の範囲に正規化
    # (x + 0.5) % 1.0 - 0.5 は、x - round(x) と等価で高速
    d_theta = (d_theta + 0.5) % 1.0 - 0.5

    # 4. プラケット周りの合計位相変化から巻き数を計算
    total_phase = np.sum(d_theta, axis=2)
    winding = np.round(total_phase).astype(int)

    # xstart, xwnd = 3, 3
    # ystart, ywnd = 1, 2
    # print(
    #     theta0[xstart : xwnd + 1, ystart : ywnd + 1],
    #     theta1[xstart : xwnd + 1, ystart : ywnd + 1],
    #     theta2[xstart : xwnd + 1, ystart : ywnd + 1],
    #     theta3[xstart : xwnd + 1, ystart : ywnd + 1],
    #     d_theta[xstart : xwnd + 1, ystart : ywnd + 1, 0],
    #     d_theta[xstart : xwnd + 1, ystart : ywnd + 1, 1],
    #     d_theta[xstart : xwnd + 1, ystart : ywnd + 1, 2],
    #     d_theta[xstart : xwnd + 1, ystart : ywnd + 1, 3],
    #     total_phase[xstart : xwnd + 1, ystart : ywnd + 1],
    #     winding[xstart : xwnd + 1, ystart : ywnd + 1],
    #     sep="\n",
    #     end="\n\n",
    # )

    # 5. 巻き数が+1または-1のプラケットのインデックスを取得
    pos_vortices_indices = np.argwhere(winding == 1)
    neg_vortices_indices = np.argwhere(winding == -1)

    # 6. 結果を返す
    if center_of_plaquette:
        positive_vortices = [(i + 0.5, j + 0.5) for i, j in pos_vortices_indices]
        negative_vortices = [(i + 0.5, j + 0.5) for i, j in neg_vortices_indices]
    else:
        positive_vortices = [tuple(pos) for pos in pos_vortices_indices]
        negative_vortices = [tuple(neg) for neg in neg_vortices_indices]

    return positive_vortices, negative_vortices


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
