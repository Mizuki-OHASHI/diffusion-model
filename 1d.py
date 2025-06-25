# 「入門物理学入門」の1D分布 (3峰) のスコアベース拡散モデル

import typing as tp

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.random.set_seed(42)
np.random.seed(42)


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
        return tf.convert_to_tensor(rv, dtype=tf.float32)

    return random_variable


def diffusion_forward(x_ini: tf.Tensor, beta_lst: np.ndarray, T: int) -> tf.Tensor:
    """
    拡散過程の前進ステップを実行する関数
    Args:
        x_ini: 初期状態のテンソル
        beta_lst: 各ステップのノイズ強度
        T: ステップ数
    """
    x = x_ini
    x_lst = [x]
    for t in range(T):
        noise = tf.random.normal(
            shape=tf.shape(x), mean=0.0, stddev=np.sqrt(beta_lst[t])
        )
        x = -1 / 2 * beta_lst[t] * x + noise
        x_lst.append(x)
    return tf.stack(x_lst, axis=0)


def diffusion_backward(
    x_fin: tf.Tensor, beta_lst: np.ndarray, score_model: keras.Model
) -> tf.Tensor:
    """
    拡散過程の後退ステップを実行する関数
    Args:
        x_fin: 最終状態のテンソル
        beta_lst: 各ステップのノイズ強度
        score_model: スコアモデル
    """
    T = len(beta_lst)
    x = x_fin
    x_lst = [x]
    for t in range(T - 1, -1, -1):
        t_arr = tf.ones_like(x) * t  # (batch, 1)
        xt = tf.concat([x, tf.cast(t_arr, tf.float32)], axis=1)  # (batch, 2)
        score = score_model(xt)
        noise = tf.random.normal(
            shape=tf.shape(x), mean=0.0, stddev=np.sqrt(beta_lst[t])
        )
        beta_t = tf.ones_like(x) * beta_lst[t]
        x = tf.multiply(tf.cast(x, tf.float32), -0.5) + beta_t * score + noise
        x_lst.append(x)
    return tf.stack(x_lst[::-1], axis=0)


# 訓練するスコアのモデル
score_model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(2,)),  # (x, t)
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ]
)

# 学習データの生成
n_input_samples = 100
mu_lst = tf.constant([-3.0, 0.0, 3.0], dtype=tf.float32)
sigma_lst = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
weight = tf.constant([0.2, 0.5, 0.3], dtype=tf.float32)
input_samples = mixed_gaussian(mu_lst, sigma_lst, weight)(n_input_samples)

# 拡散過程のパラメータ
beta_lst = tf.linspace(0.01, 0.1, 100)
T = beta_lst.shape[0]

# 拡散過程の前進ステップを実行
x_lst = diffusion_forward(input_samples, beta_lst, T)

beta_tensor = beta_lst

for epoch in tqdm(range(1000)):
    idx = tf.random.uniform([n_input_samples], minval=0, maxval=T, dtype=tf.int32)
    gather_idx = tf.stack([idx, tf.range(n_input_samples, dtype=tf.int32)], axis=1)
    x_t = tf.gather_nd(x_lst, gather_idx)  # (n,1)
    t = tf.cast(idx, tf.float32)
    xt = tf.concat(
        [tf.reshape(x_t, [n_input_samples, 1]), tf.reshape(t, [n_input_samples, 1])],
        axis=1,
    )  # (n,2)
    xt_tensor = tf.convert_to_tensor(xt, dtype=tf.float32)
    with tf.GradientTape() as tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(xt_tensor)
            score = score_model(xt_tensor)
        score_grad = inner_tape.gradient(score, xt_tensor)
        score_div = tf.reduce_sum(score_grad, axis=-1)
        score_squared = tf.square(score_div)
        loss_terms = score_squared + 2.0 * score_div
        beta_batch = tf.gather(beta_tensor, idx)
        loss = tf.reduce_mean(beta_batch * loss_terms)
    gradients = tape.gradient(loss, score_model.trainable_variables)
    if gradients is not None:
        keras.optimizers.Adam(learning_rate=0.001).apply_gradients(
            zip(gradients, score_model.trainable_variables)
        )

# モデルの保存
score_model.save("models/1d_score_model.h5")

# モデルの読み込み
loaded_model = keras.models.load_model("models/1d_score_model.h5")

# 拡散過程の後退ステップを実行
x_fin = tf.random.normal(shape=(1000, 1), mean=0.0, stddev=1.0)  # 漸近分布
x_reconstructed = diffusion_backward(x_fin, beta_lst, loaded_model)  # type: ignore

# 結果の表示
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].hist(input_samples, bins=50, density=True, alpha=0.5, label="Input Samples")
axes[0].set_title("Input Samples Distribution")
axes[0].legend()
axes[1].hist(
    x_reconstructed[-1], bins=50, density=True, alpha=0.5, label="Reconstructed Samples"  # type: ignore
)
axes[1].set_title("Reconstructed Samples Distribution")
axes[1].legend()
plt.tight_layout()
plt.show()
