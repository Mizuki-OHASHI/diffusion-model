import hashlib
import json
import os
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


def hash_dict(d: dict) -> str:
    # ソートして順番に依存しないようにする
    d_str = json.dumps(d, sort_keys=True)
    # ハッシュ化して16進数で返す（SHA256 -> 64文字だが短くしてもOK）
    return hashlib.sha256(d_str.encode()).hexdigest()[:16]


def save_np(data: np.ndarray, input: dict[str, str | int | float]) -> None:
    """
    NumPy配列をファイルに保存する関数

    Parameters
    ----------
    data : np.ndarray
        保存するNumPy配列
    input : dict[str, str | int | float]
        入力パラメータを含む辞書。データを識別するために使用します。
        同じ入力パラメータであれば、同じファイル名で保存されます。
    """
    # input からハッシュ生成
    hash_key = hash_dict(input)
    filename = f"cache/data_{hash_key}.npy"
    # ディレクトリが存在しない場合は作成
    if not os.path.exists("cache"):
        os.makedirs("cache")
    np.save(filename, data)
    print(f"Saved data to {filename}")


def load_np(input: dict[str, str | int | float]) -> np.ndarray | None:
    """
    NumPy配列をファイルから読み込む関数
    """
    # input からハッシュ生成
    hash_key = hash_dict(input)
    filename = f"cache/data_{hash_key}.npy"
    if os.path.exists(filename):
        data = np.load(filename)
        print(f"Loaded data from {filename}")
        return data
    else:
        print(f"No data found for {filename}")
        return None
