# 2次元Isingモデル (メトロポリス法)
# H = - J Sum_<i,j> s_i s_j - h Sum_i s_i

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from onsager import (
    onsager_energy,
    onsager_heat_capacity,
    onsager_magnetization,
    theoretical_critical_kT,
)
from utils import load_np, save_np

np.random.seed(42)


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
    n_mcs_equilibration: int,
    n_mcs_measurement: int,
    measurement_interval_mcs: int,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    純粋なメトロポリス法によるサンプリング (MCSベース)
    """
    x = x_ini.copy()
    L = x.shape[0]
    n_spins = L * L
    energy_samples = []
    magnetization_samples = []
    samples = []

    # 平衡化ステップ (MCS単位)
    iterator = range(n_mcs_equilibration)
    if progress:
        iterator = tqdm(iterator, desc="Equilibration (MCS)")
    for _ in iterator:
        for _ in range(n_spins):  # 1 MCS
            x = pure_metropolis_next(x, beta, J, h)

    # 測定ステップ (MCS単位)
    iterator = range(n_mcs_measurement)
    if progress:
        iterator = tqdm(iterator, desc="Measurement (MCS)")
    for i in iterator:
        for _ in range(n_spins):  # 1 MCS
            x = pure_metropolis_next(x, beta, J, h)
        if i % measurement_interval_mcs == 0:
            energy = ising_energy(x[np.newaxis, ...], J, h)[0]
            magnetization = ising_magnetization(x[np.newaxis, ...])[0]
            energy_samples.append(energy)
            magnetization_samples.append(magnetization)
            samples.append(x.copy())

    return np.array(samples), np.array(energy_samples), np.array(magnetization_samples)


# パラメータ
J = 1.0  # 相互作用定数
h = 0.0  # 外部磁場
L = 20  # 格子の一辺の長さ
N_spins = L * L

# モンテカルロステップ (MCS) ベースのパラメータ設定
n_mcs_equilibration = 1000  # 平衡化のMCS
n_mcs_measurement = 5000  # 測定のMCS
measurement_interval_mcs = 10  # 測定間隔 (MCS)


def metropolis_worker(beta, J, h, L):
    x_ini = np.ones((L, L), dtype=int)
    energy_samples, magnetization_samples = pure_metropolis(
        x_ini,
        beta,
        J,
        h,
        n_mcs_equilibration,
        n_mcs_measurement,
        measurement_interval_mcs,
        # progress=False,
    )
    return energy_samples, magnetization_samples


def pure_metropolis_run():
    """
    純粋なメトロポリス法によるIsingモデルのサンプリングを実行
    - 温度依存性を調べる
    """
    kT_lst = np.linspace(0.1, 6.0, 100)
    beta_lst = 1 / kT_lst

    # ファイル名にMCSの情報を追加
    param_str = f"L{L}_eq{n_mcs_equilibration}_mc{n_mcs_measurement}"
    input_dict_energy = dict(
        name=f"pure_metropolis_run.energy.{param_str}",
    )
    input_dict_magnetization = dict(
        name=f"pure_metropolis_run.magnetization.{param_str}",
    )

    energy_samples_lst = load_np(input_dict_energy)
    magnetization_samples_lst = load_np(input_dict_magnetization)

    if energy_samples_lst is None or magnetization_samples_lst is None:
        results = []
        for beta in tqdm(beta_lst, desc="Running Pure Metropolis for each temperature"):
            r = metropolis_worker(beta, J, h, L)
            results.append(r)
        energy_samples_lst, magnetization_samples_lst = zip(*results)
        energy_samples_lst = np.array(
            [np.array(e) for e in energy_samples_lst], dtype=object
        )
        magnetization_samples_lst = np.array(
            [np.array(m) for m in magnetization_samples_lst], dtype=object
        )

        # データの保存
        save_np(energy_samples_lst, input_dict_energy)
        save_np(magnetization_samples_lst, input_dict_magnetization)

    # 各温度での物理量の平均と標準偏差を計算
    energy_mean = np.array([np.mean(e) for e in energy_samples_lst]) / N_spins
    energy_std = np.array([np.std(e) for e in energy_samples_lst]) / N_spins
    magnetization_mean = np.array(
        [np.mean(np.abs(m)) for m in magnetization_samples_lst]
    )
    magnetization_std = np.array([np.std(m) for m in magnetization_samples_lst])

    # 比熱 (揺らぎの公式)
    # C/N = (<E^2> - <E>^2) / (N * kT^2)
    energy_var = np.array([np.var(e) for e in energy_samples_lst])
    heat_capacity = energy_var / (N_spins * kT_lst**2)

    # Onsagerの厳密解
    onsager_kT = np.linspace(0.1, 6.0, 200)
    onsager_E = onsager_energy(onsager_kT, J)
    onsager_M = onsager_magnetization(onsager_kT, J)
    onsager_C = onsager_heat_capacity(onsager_kT, J)

    # 結果のプロット
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Pure Metropolis Sampling of 2D Ising Model (L={L}, Eq={n_mcs_equilibration} MCS, Mc={n_mcs_measurement} MCS)",
        fontsize=16,
    )

    # エネルギー
    ax[0].errorbar(
        kT_lst,
        energy_mean,
        yerr=energy_std,
        fmt="o",
        label="Metropolis",
        color="blue",
        markersize=4,
        capsize=3,
    )
    ax[0].plot(onsager_kT, onsager_E, label="Onsager (Exact)", color="red")
    ax[0].axvline(
        theoretical_critical_kT, color="gray", linestyle="--", label="Critical Temp."
    )
    ax[0].set_title("Energy vs Temperature")
    ax[0].set_xlabel("Temperature (kT/J)")
    ax[0].set_ylabel("Energy / N")
    ax[0].legend()

    # 磁化
    ax[1].errorbar(
        kT_lst,
        magnetization_mean,
        yerr=magnetization_std,
        fmt="o",
        label="Metropolis",
        color="orange",
        markersize=4,
        capsize=3,
    )
    ax[1].plot(onsager_kT, onsager_M, label="Onsager (Exact)", color="red")
    ax[1].axvline(theoretical_critical_kT, color="gray", linestyle="--")
    ax[1].set_title("Magnetization vs Temperature")
    ax[1].set_xlabel("Temperature (kT/J)")
    ax[1].set_ylabel("Magnetization / N")
    ax[1].legend()

    # 比熱
    ax[2].plot(
        kT_lst, heat_capacity, "o-", label="Metropolis", color="green", markersize=4
    )
    ax[2].plot(onsager_kT, onsager_C, label="Onsager (Exact)", color="red")
    ax[2].axvline(theoretical_critical_kT, color="gray", linestyle="--")
    ax[2].set_title("Specific Heat vs Temperature")
    ax[2].set_xlabel("Temperature (kT/J)")
    ax[2].set_ylabel("Specific Heat / N")
    ax[2].legend()
    ax[2].set_ylim(bottom=0)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    # plt.show()
    plt.savefig(
        f"figures/pure_metropolis_2d_ising_{L}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    pure_metropolis_run()
