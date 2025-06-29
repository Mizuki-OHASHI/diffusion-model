# 2次元Isingモデル
# H = - J Sum_<i,j> s_i s_j - h Sum_i s_i

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe, ellipk
from tqdm import tqdm

from utils import load_np, save_np

# import tensorflow as tf

np.random.seed(42)

theoretical_critical_kT = 2.2691  # 2次元Isingモデルの理論的臨界温度 (kT)


def onsager_energy(kT, J):
    """
    Onsager's solution for the energy per site of the 2D Ising model.
    """
    beta = 1.0 / kT
    k = 2 * np.sinh(2 * beta * J) / (np.cosh(2 * beta * J) ** 2)
    K = ellipk(k**2)
    E_N = (
        -J
        * (1.0 / np.tanh(2 * beta * J))
        * (1 + (2.0 / np.pi) * (2 * np.tanh(2 * beta * J) ** 2 - 1) * K)
    )
    return E_N


def onsager_magnetization(kT, J):
    """
    Onsager's solution for spontaneous magnetization per site of 2D Ising model.
    """
    m = np.zeros_like(kT)
    kTc = 2 * J / np.arcsinh(1.0)
    with np.errstate(invalid="ignore"):
        magnetization = (1 - np.sinh(2 * J / kT) ** (-4)) ** (1 / 8.0)
    m[kT < kTc] = magnetization[kT < kTc]
    return m


def onsager_heat_capacity(kT, J):
    """
    Onsager's solution for the heat capacity per site of the 2D Ising model.
    """
    beta = 1.0 / kT
    with np.errstate(invalid="ignore"):
        k = 2 * np.sinh(2 * beta * J) / (np.cosh(2 * beta * J) ** 2)
        K = ellipk(k**2)
        E = ellipe(k**2)
        k_prime = 2 * np.tanh(2 * beta * J) ** 2 - 1
        # C/k_B
        C_N_kB = (
            (2.0 / np.pi)
            * (2.0 * J * beta / np.tanh(2 * beta * J)) ** 2
            * (2.0 * K - 2.0 * E - (1.0 - k_prime) * (np.pi / 2.0 + k_prime * K))
        )
    return C_N_kB


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
L = 50  # 格子の一辺の長さ
n_steps = 100000  # サンプリングステップ数
beta = 0.1  # 逆温度


def metropolis_worker(beta, J, h, L, n_steps):
    x_ini = np.ones((L, L), dtype=int)
    x_lst = pure_metropolis(x_ini, beta, J, h, n_steps, progress=False)
    energy = ising_energy(x_lst, J, h)
    magnetization = ising_magnetization(x_lst)
    return energy, magnetization


def pure_metropolis_run():
    """
    純粋なメトロポリス法によるIsingモデルのサンプリングを実行
    - 温度依存性を調べる
    """
    kT_lst = np.linspace(0.1, 6.0, 100)
    beta_lst = 1 / kT_lst

    input_dict_energy = dict(
        name="pure_metropolis_run.energy",
        L=L,
        n_steps=n_steps,
        J=J,
        h=h,
        beta=str(beta_lst),
    )
    input_dict_magnetization = dict(
        name="pure_metropolis_run.magnetization",
        L=L,
        n_steps=n_steps,
        J=J,
        h=h,
        beta=str(beta_lst),
    )

    energy_lst = load_np(input_dict_energy)
    magnetization_lst = load_np(input_dict_magnetization)
    if energy_lst is None or magnetization_lst is None:
        results = []
        for beta in tqdm(beta_lst, desc="Running Pure Metropolis"):
            r = metropolis_worker(beta, J, h, L, n_steps)
            results.append(r)
        energy_lst, magnetization_lst = zip(*results)
        energy_lst = np.array(energy_lst)
        magnetization_lst = np.array(magnetization_lst)

        # データの保存
        save_np(energy_lst, input_dict_energy)
        save_np(magnetization_lst, input_dict_magnetization)

    # エネルギー・磁化の平均と標準偏差 (後半の50%のデータを使用)
    equil_step = n_steps // 2
    energy_mean = np.mean(energy_lst[:, -equil_step:], axis=1) / (L * L)
    energy_std = np.std(energy_lst[:, -equil_step:], axis=1) / (L * L)
    magnetization_mean = np.mean(np.abs(magnetization_lst[:, -equil_step:]), axis=1)
    magnetization_std = np.std(magnetization_lst[:, -equil_step:], axis=1)

    # 比熱 (C = dE/dT)
    # heat_capacity = np.gradient(energy_mean, kT_lst)
    # 揺らぎの公式を使って比熱を計算
    # C/N = (<E^2> - <E>^2) / (N * kT^2)
    heat_capacity = np.var(energy_lst[:, -equil_step:], axis=1) / (kT_lst**2 * (L * L))

    # Onsagerの厳密解
    onsager_kT = np.linspace(0.1, 6.0, 200)
    onsager_E = onsager_energy(onsager_kT, J)
    onsager_M = onsager_magnetization(onsager_kT, J)
    onsager_C = onsager_heat_capacity(onsager_kT, J)

    # 結果のプロット
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Pure Metropolis Sampling of 2D Ising Model (L = {L})".format(L=L), fontsize=16
    )
    # 代表点でのエネルギー by iteration
    idx = np.arange(0, len(kT_lst), len(kT_lst) // 5)
    for i in idx:
        ax[0, 0].plot(
            np.arange(n_steps + 1),
            energy_lst[i] / (L * L),
            label=f"kT = {kT_lst[i]:.2f}",
        )
    ax[0, 0].set_title("Energy vs Iteration")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Energy / N")
    ax[0, 0].legend()
    # エネルギー
    ax[0, 1].errorbar(
        kT_lst,
        energy_mean,
        yerr=energy_std,
        fmt="o",
        label="Metropolis",
        color="blue",
        markersize=4,
    )
    ax[0, 1].plot(onsager_kT, onsager_E, label="Onsager", color="red")
    ax[0, 1].axvline(theoretical_critical_kT, color="red", linestyle="--")
    ax[0, 1].set_title("Energy vs Temperature")
    ax[0, 1].set_xlabel("Temperature (kT)")
    ax[0, 1].set_ylabel("Energy / N")
    ax[0, 1].legend()
    # 磁化
    ax[1, 0].errorbar(
        kT_lst,
        magnetization_mean,
        yerr=magnetization_std,
        fmt="o",
        label="Metropolis",
        color="orange",
        markersize=4,
    )
    ax[1, 0].plot(onsager_kT, onsager_M, label="Onsager", color="red")
    ax[1, 0].axvline(theoretical_critical_kT, color="red", linestyle="--")
    ax[1, 0].set_title("Magnetization vs Temperature")
    ax[1, 0].set_xlabel("Temperature (kT)")
    ax[1, 0].set_ylabel("Magnetization / N")
    ax[1, 0].legend()
    # 比熱
    ax[1, 1].plot(kT_lst, heat_capacity, "o-", label="Metropolis", color="green")
    ax[1, 1].plot(onsager_kT, onsager_C, label="Onsager", color="red")
    ax[1, 1].axvline(theoretical_critical_kT, color="red", linestyle="--")
    ax[1, 1].set_title("Heat Capacity vs Temperature")
    ax[1, 1].set_ylim(None, 1.1 * np.max(heat_capacity))
    ax[1, 1].set_xlabel("Temperature (kT)")
    ax[1, 1].set_ylabel("Heat Capacity / N")
    ax[1, 1].legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "figures/pure_metropolis_2d_ising_{L}.png".format(L=L),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    pure_metropolis_run()
