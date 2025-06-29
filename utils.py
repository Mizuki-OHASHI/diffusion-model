import typing as tp

import numpy as np
import tensorflow as tf


def mixed_gaussian(mu_lst, sigma_lst, weight) -> tp.Callable[[int], tf.Tensor]:
    """
    混合ガウス分布に従う乱数を生成する関数
    """
    length = len(mu_lst)
    assert length == len(sigma_lst) == len(weight)
    mu_lst = np.array(mu_lst, dtype=np.float32)
    sigma_lst = np.array(sigma_lst, dtype=np.float32)
    weight = np.array(weight, dtype=np.float32)

    def random_variable(n_samples: int) -> tf.Tensor:
        index = np.random.choice(length, size=n_samples, p=weight)
        mu = mu_lst[index]
        sigma = sigma_lst[index]
        rv = np.random.normal(mu, sigma, size=n_samples)
        return tf.convert_to_tensor(rv[:, np.newaxis], dtype=tf.float32)

    return random_variable


def mixed_gaussian_pdf(
    mu_lst, sigma_lst, weight
) -> tp.Callable[[np.ndarray], np.ndarray]:
    """
    混合ガウス分布の確率密度関数を計算する関数
    """
    length = len(mu_lst)
    assert length == len(sigma_lst) == len(weight)
    mu_lst = np.array(mu_lst, dtype=np.float32)
    sigma_lst = np.array(sigma_lst, dtype=np.float32)
    weight = np.array(weight, dtype=np.float32)

    def pdf(x: np.ndarray) -> np.ndarray:
        pdf_values = np.zeros_like(x, dtype=np.float32)
        for mu, sigma, w in zip(mu_lst, sigma_lst, weight):
            pdf_values += (
                w
                * (1 / (sigma * np.sqrt(2 * np.pi)))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )
        return pdf_values

    return pdf
