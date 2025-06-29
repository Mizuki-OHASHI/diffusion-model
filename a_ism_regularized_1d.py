# 「入門物理学入門」の1D分布 (3峰) のスコアベース拡散モデル
# ISM: Implicit Score Matching 暗黙的スコアマッチング (正則化あり)


import japanize_matplotlib  # noqa: F401
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import mixed_gaussian, mixed_gaussian_pdf

tf.random.set_seed(42)
np.random.seed(42)


def diffusion_forward(x_ini: tf.Tensor, beta_lst: np.ndarray, T: int) -> tf.Tensor:
    """
    拡散過程の前進ステップを実行する関数
    """
    x = x_ini
    x_lst = [x]
    for t in range(T):
        noise = tf.random.normal(
            shape=tf.shape(x), mean=0.0, stddev=tf.sqrt(beta_lst[t])
        )
        x = (1 - 0.5 * beta_lst[t]) * x + noise
        x_lst.append(x)
    return tf.stack(x_lst, axis=0)


def diffusion_backward(
    x_fin: tf.Tensor, beta_lst: np.ndarray, score_model: keras.Model
) -> tf.Tensor:
    """
    拡散過程の後退ステップを実行する関数
    """
    T = len(beta_lst)
    x = x_fin
    x_lst = [x]
    for t in range(T - 1, -1, -1):
        t_arr = tf.ones_like(x) * t / tf.cast(T, tf.float32)
        xt = tf.concat([x, tf.cast(t_arr, tf.float32)], axis=1)  # (batch, 2)
        score = score_model(xt)
        noise = tf.random.normal(
            shape=tf.shape(x), mean=0.0, stddev=tf.sqrt(beta_lst[t])
        )
        beta_t = beta_lst[t]

        x = (1 + 0.5 * beta_t) * x + beta_t * score + noise
        x_lst.append(x)
    return tf.stack(x_lst[::-1], axis=0)


# 訓練するスコアのモデル
score_model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(2,), name="input_layer"),  # (x, t)
        keras.layers.Dense(16, activation="tanh", name="dense_1"),
        keras.layers.Dense(32, activation="tanh", name="dense_2"),
        keras.layers.Dense(16, activation="tanh", name="dense_3"),
        keras.layers.Dense(8, activation="tanh", name="dense_4"),
        keras.layers.Dense(1, activation="linear", name="output_layer"),
    ]
)
score_model.summary()

# 学習データの生成
n_input_samples = 2000
mu_lst = tf.constant([-2.0, 0.0, 2.0], dtype=tf.float32)
sigma_lst = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
weight = tf.constant([0.25, 0.6, 0.15], dtype=tf.float32)
input_samples = mixed_gaussian(mu_lst, sigma_lst, weight)(n_input_samples)


# 拡散過程のパラメータ
beta_lst = tf.linspace(0.01, 0.1, 100).numpy()  # numpy配列に変換
T = beta_lst.shape[0]
regularization_strength = 1e-2  # 正則化強度

# 拡散過程の前進ステップを実行
print("Running forward diffusion process...")
x_lst = diffusion_forward(input_samples, beta_lst, T)
print("Forward process finished.")

optimizer = keras.optimizers.Adam(learning_rate=0.001)

# beta_lstをTensorに変換
beta_tensor = tf.convert_to_tensor(beta_lst, dtype=tf.float32)


@tf.function
def train_step(score_model: keras.Model, xt_tensor: tf.Tensor) -> None:
    with tf.GradientTape() as tape:
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch(xt_tensor)
            # モデルからスコアを予測
            predicted_score = score_model(xt_tensor)

        grad_s_xt = inner_tape.gradient(predicted_score, xt_tensor)
        score_divergence = tf.gather(grad_s_xt, 0, axis=1)
        score_norm_squared = tf.square(tf.squeeze(predicted_score))

        # スコアマッチングの損失
        # J(θ) = E[ tr(∇_x s(x,t)) + (1/2) * ||s(x,t)||^2 ]
        loss_terms = (
            score_divergence
            + 0.5 * score_norm_squared
            + regularization_strength * tf.square(predicted_score)
        )
        loss = tf.reduce_mean(loss_terms)

        # persistent=True のテープは手動で解放する必要がある
        del inner_tape

    gradients = tape.gradient(loss, score_model.trainable_variables)
    if gradients is not None:
        # ループの外で定義したオプティマイザを使用
        optimizer.apply_gradients(zip(gradients, score_model.trainable_variables))


for epoch in tqdm(range(5000), desc="Training"):
    idx = tf.random.uniform([n_input_samples], minval=0, maxval=T, dtype=tf.int32)
    gather_idx = tf.stack([idx, tf.range(n_input_samples, dtype=tf.int32)], axis=1)
    x_t = tf.gather_nd(x_lst, gather_idx)
    t = tf.cast(idx, tf.float32)
    t_norm = tf.divide(t, tf.cast(T, tf.float32))

    # モデルへの入力 (x_t, t_norm)
    xt = tf.concat(
        [
            tf.reshape(x_t, [n_input_samples, 1]),
            tf.reshape(t_norm, [n_input_samples, 1]),
        ],
        axis=1,
    )
    xt_tensor = tf.convert_to_tensor(xt, dtype=tf.float32)
    train_step(score_model, xt_tensor)


# モデルの保存
score_model.save("models/1d_ism_regularized_score_model.keras")

# モデルの読み込み
loaded_model = keras.models.load_model("models/1d_ism_regularized_score_model.keras")

# 最後のステップのデータを取得
# x_fin = np.array(x_lst)[-1].reshape(-1, 1)
# x_fin = tf.convert_to_tensor(x_fin, dtype=tf.float32)
x_fin = tf.random.normal(shape=(n_input_samples, 1), mean=0.0, stddev=1.0)

# 拡散過程の後退ステップを実行
x_asymp = tf.random.normal(shape=tf.shape(x_fin), mean=0.0, stddev=1.0)
x_reconstructed = diffusion_backward(x_asymp, beta_lst, loaded_model)  # type: ignore

# 結果の表示
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
bins = np.linspace(-4, 4, 40)

pdf = mixed_gaussian_pdf(mu_lst, sigma_lst, weight)
x_range = np.linspace(-4, 4, 1000)

# 入力データと拡散後の分布
axes[0].plot(x_range, pdf(x_range), "b--", label="確率密度関数", linewidth=2)
axes[0].hist(
    np.array(input_samples).flatten(),
    bins=bins,
    density=True,
    alpha=0.6,
    label="学習データ",
    color="blue",
)
axes[0].hist(
    np.array(x_reconstructed)[-1].flatten(),
    bins=bins,
    density=True,
    alpha=0.6,
    label="拡散後の分布",
    color="red",
)
axes[0].set_title("入力データと拡散後の分布")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Density")
axes[0].legend(loc="upper right")
axes[0].set_xlim(-4, 4)

# 訓練済みスコアで逆拡散した結果
xr_last = np.array(x_reconstructed)[0].flatten()
axes[1].plot(x_range, pdf(x_range), "b--", label="確率密度関数", linewidth=2)
axes[1].hist(
    xr_last,
    bins=bins,
    density=True,
    alpha=0.6,
    label="再構成されたサンプル",
    color="blue",
)
axes[1].set_title("訓練済みスコアで逆拡散した結果")
axes[1].set_xlabel("Value")
axes[1].legend(loc="upper right")

plt.tight_layout()
# plt.show()
plt.savefig("figures/1d_ism_regularized_result.png", dpi=300, bbox_inches="tight")
plt.close(fig)
