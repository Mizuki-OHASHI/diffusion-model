# 「入門物理学入門」の1D分布 (3峰) のスコアベース拡散モデル
# DSM: Denoising Score Matching デノイジングスコアマッチング


import japanize_matplotlib  # noqa: F401
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import mixed_gaussian, mixed_gaussian_pdf

tf.random.set_seed(42)
np.random.seed(42)


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

        # x = (1 + 0.5 * beta_t) * x + beta_t * score + noise
        x = 1 / tf.sqrt(1 - beta_t) * x + beta_t / tf.sqrt(1 - beta_t) * score + noise
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
beta_lst = tf.linspace(0.01, 0.1, 1000).numpy()  # numpy配列に変換
alpha_lst = 1 - beta_lst
alpha_cumprod = np.cumprod(alpha_lst)
T = beta_lst.shape[0]

optimizer = keras.optimizers.Adam(learning_rate=0.001)

alpha_cumprod_tensor = tf.convert_to_tensor(alpha_cumprod, dtype=tf.float32)


@tf.function
def train_step(
    score_model: keras.Model, x_initial: tf.Tensor, t_values: tf.Tensor, T: int
) -> None:
    with tf.GradientTape() as tape:
        # 摂動カーネルのノイズスケジュール
        # sigma_t = tf.sqrt(tf.gather(beta_tensor, t_values))
        alpha_t = tf.gather(alpha_cumprod_tensor, t_values)

        # ノイズを加える
        noise = tf.random.normal(
            shape=tf.shape(x_initial), mean=0.0, stddev=tf.sqrt(1 - alpha_t)[:, None]
        )
        x_t = tf.sqrt(alpha_t[:, None]) * x_initial + noise

        # tを正規化
        t_norm = tf.divide(tf.cast(t_values, tf.float32), tf.cast(T, tf.float32))

        # モデルへの入力 (x_t, t_norm)
        xt_input = tf.concat([x_t, t_norm[:, None]], axis=1)

        # モデルからスコアを予測
        predicted_score = score_model(xt_input)

        # ターゲットスコア
        target_score = -noise / (1 - alpha_t)[:, None]

        # Denoising Score Matchingの損失
        loss = tf.reduce_mean(tf.square(predicted_score - target_score))

    gradients = tape.gradient(loss, score_model.trainable_variables)
    if gradients is not None:
        optimizer.apply_gradients(zip(gradients, score_model.trainable_variables))


batch_size = 128
n_epochs = 50000
for epoch in tqdm(range(n_epochs), desc="Training"):
    sample_idx = tf.random.uniform(
        [batch_size], minval=0, maxval=n_input_samples, dtype=tf.int32
    )
    input_samples_batch = tf.gather(input_samples, sample_idx)
    train_step(
        score_model,
        input_samples_batch,
        tf.random.uniform([batch_size], 0, T, dtype=tf.int32),
        T,
    )


# モデルの保存
score_model.save("models/1d_dsm_score_model.keras")

# モデルの読み込み
loaded_model = keras.models.load_model("models/1d_dsm_score_model.keras")

# 最後のステップのデータを取得
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
# axes[0].hist(
#     np.array(x_reconstructed)[-1].flatten(),
#     bins=bins,
#     density=True,
#     alpha=0.6,
#     label="拡散後の分布",
#     color="red",
# )
axes[0].set_title("入力データ")
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
plt.savefig("figures/1d_dsm_result.png", dpi=300, bbox_inches="tight")
plt.close(fig)
