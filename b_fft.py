import numpy as np


def ising_fft(x_lst: np.ndarray, L: int) -> np.ndarray:
    """
    2次元のスピン配置をフーリエ変換する関数
    """
    # 2次元のフーリエ変換を適用
    return np.fft.rfft2(x_lst, s=(L, L))


def ising_ifft(x_lst: np.ndarray, L: int) -> np.ndarray:
    """
    2次元のフーリエ逆変換を適用する関数
    """
    # 2次元のフーリエ逆変換を適用
    return np.fft.irfft2(x_lst, s=(L, L))


if __name__ == "__main__":
    L = 10
    samples = np.random.randint(0, 2, size=(1, L, L)) * 2 - 1
    fft_samples = np.random.randn(1, L, L) + 1j * np.random.randn(1, L, L)

    print("Original Spin Configuration:")
    print(samples)
    print("\nFourier Transform of Spin Configuration:")
    fft_result = ising_fft(samples, L)
    print(fft_result.round(0))
    print("\nInverse Fourier Transform (Reconstructed Spin Configuration):")
    reconstructed_sample = ising_ifft(fft_result, L)
    print(reconstructed_sample)
    print("\nReconstructed Spin Configuration (rounded):")
    print(np.where(ising_ifft(fft_samples, L) > 0, 1, -1))
