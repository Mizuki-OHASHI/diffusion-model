import numpy as np
from scipy.special import ellipe, ellipk

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
