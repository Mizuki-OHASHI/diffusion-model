# 2次元Isingモデル
# H = - J Sum_<i,j> s_i s_j - h Sum_i s_i

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import tensorflow as tf

np.random.seed(42)

theoretical_critical_kT = 2.2691  # 2次元Isingモデルの理論的臨界温度 (kT)


def pure_metropolis_next(x: np.ndarray, beta: float, J: float, h: float) -> np.ndarray:
    """
    純粋なメトロポリス法によるサンプリング
    """
    # 反転するスピンのインデックスをランダムに選択
    idx = np.random.randint(0, x.size)
    i, j = divmod(idx, x.shape[1])
    s = x[i, j]  # 選択したスピンの値
    # エネルギーの変化を計算 (周期境界条件)
    delta_nn = (
        x[(i + 1) % x.shape[0], j]
        + x[i, (j + 1) % x.shape[1]]
        + x[(i - 1) % x.shape[0], j]
        + x[i, (j - 1) % x.shape[1]]
    )
    delta_E = 2 * s * (J * delta_nn + h)
    # メトロポリス条件を適用
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        x[i, j] = -s  # スピンを反転
    return x


def ising_energy(x_lst: np.ndarray, J: float, h: float) -> np.ndarray:
    """
    Isingモデルのエネルギーを計算 (周期境界条件)
    H = - J Sum_<i,j> s_i s_j - h Sum_i s_i
    """
    # 相互作用を計算 (周期境界条件)
    interaction_vert = np.sum(x_lst * np.roll(x_lst, shift=-1, axis=1), axis=(1, 2))
    interaction_horiz = np.sum(x_lst * np.roll(x_lst, shift=-1, axis=2), axis=(1, 2))
    interaction = interaction_vert + interaction_horiz
    field = np.sum(x_lst, axis=(1, 2))
    energy = -J * interaction - h * field
    return energy


def ising_magnetization(x_lst: np.ndarray) -> np.ndarray:
    """
    Isingモデルの磁化を計算
    """
    return np.mean(x_lst, axis=(1, 2))


def pure_metropolis(
    x_ini: np.ndarray,
    beta: float,
    J: float,
    h: float,
    n_steps: int,
    progress: bool = True,
) -> np.ndarray:
    """
    純粋なメトロポリス法によるサンプリング
    """
    x = x_ini.copy()
    x_lst = [x]
    for _ in tqdm(
        range(n_steps), desc="Pure Metropolis Sampling", disable=not progress
    ):
        x = pure_metropolis_next(x, beta, J, h)
        x_lst.append(x.copy())
    return np.array(x_lst)


# パラメータ
J = 1.0  # 相互作用定数
h = 0.0  # 外部磁場
L = 20  # 格子の一辺の長さ
n_steps = 50000  # サンプリングステップ数
beta = 0.1  # 逆温度


def pure_metropolis_run():
    """
    純粋なメトロポリス法によるIsingモデルのサンプリングを実行
    - 温度依存性を調べる
    """
    kT_lst = np.linspace(0.1, 6.0, 100)
    beta_lst = 1 / kT_lst
    energy_lst = []
    magnetization_lst = []
    for beta in tqdm(beta_lst, desc="Running Pure Metropolis"):
        # 初期状態の生成
        # x_ini = np.random.choice([-1, 1], size=(L, L))  # ランダムなスピン配置
        x_ini = np.ones((L, L), dtype=int)  # 全てのスピンを+1に初期化

        # サンプリング
        x_lst = pure_metropolis(x_ini, beta, J, h, n_steps, progress=False)

        # エネルギーと磁化の計算
        energy = ising_energy(x_lst, J, h)
        magnetization = ising_magnetization(x_lst)

        energy_lst.append(energy)
        magnetization_lst.append(magnetization)

    energy_lst = np.array(energy_lst)
    magnetization_lst = np.array(magnetization_lst)

    # エネルギー・磁化の平均と標準偏差 (最後の10%のデータを使用)
    energy_mean = np.mean(energy_lst[:, -n_steps // 10 :], axis=1)
    energy_std = np.std(energy_lst[:, -n_steps // 10 :], axis=1)
    magnetization_mean = np.mean(magnetization_lst[:, -n_steps // 10 :], axis=1)
    magnetization_std = np.std(magnetization_lst[:, -n_steps // 10 :], axis=1)

    # 比熱 (C = dE/dT)
    heat_capacity = np.gradient(energy_mean, kT_lst)

    # 結果のプロット
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pure Metropolis Sampling of 2D Ising Model (L = 20)", fontsize=16)
    # 代表点でのエネルギー by iteration
    idx = np.arange(0, len(kT_lst), len(kT_lst) // 5)
    for i in idx:
        ax[0, 0].plot(
            np.arange(n_steps + 1), energy_lst[i], label=f"kT = {kT_lst[i]:.2f}"
        )
    ax[0, 0].set_title("Energy vs Iteration")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Energy")
    ax[0, 0].legend()
    # エネルギー
    ax[0, 1].errorbar(
        kT_lst, energy_mean, yerr=energy_std, fmt="o-", label="Energy", color="blue"
    )
    ax[0, 1].axvline(theoretical_critical_kT, color="red", linestyle="--")
    ax[0, 1].set_title("Energy vs Temperature")
    ax[0, 1].set_xlabel("Temperature (kT)")
    ax[0, 1].set_ylabel("Energy")
    ax[0, 1].legend()
    # 磁化
    ax[1, 0].errorbar(
        kT_lst,
        magnetization_mean,
        yerr=magnetization_std,
        fmt="o-",
        label="Magnetization",
        color="orange",
    )
    ax[1, 0].axvline(theoretical_critical_kT, color="red", linestyle="--")
    ax[1, 0].set_title("Magnetization vs Temperature")
    ax[1, 0].set_xlabel("Temperature (kT)")
    ax[1, 0].set_ylabel("Magnetization")
    ax[1, 0].legend()
    # 比熱
    ax[1, 1].plot(kT_lst, heat_capacity, label="Heat Capacity", color="green")
    ax[1, 1].axvline(theoretical_critical_kT, color="red", linestyle="--")
    ax[1, 1].set_title("Heat Capacity vs Temperature")
    ax[1, 1].set_xlabel("Temperature (kT)")
    ax[1, 1].set_ylabel("Heat Capacity")
    ax[1, 1].legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("figures/pure_metropolis_2d_ising.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pure_metropolis_run()
