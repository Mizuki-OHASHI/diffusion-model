# 2 次元イジングモデル (拡散モデル)

import japanize_matplotlib  # noqa: F401
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from b_pure_metoropolis_2d_ising import pure_metropolis
from utils import load_np, save_np

np.random.seed(42)
tf.random.set_seed(42)

# Ising Model Parameters
L = 20
kT = 2.0  # 温度
J = 1.0  # 相互作用定数
h = 0.0  # 外部磁場


def diffusion_backward(
    x_fin: tf.Tensor, beta_lst: np.ndarray, score_model: keras.Model
) -> tf.Tensor:
    T = len(beta_lst)
    x = x_fin
    x_lst = [x]
    for t in range(T - 1, -1, -1):
        t_arr = tf.ones([tf.shape(x)[0], 1]) * (t / T)  # type: ignore
        xt = tf.concat([x, tf.cast(t_arr, tf.float32)], axis=1)
        score = score_model(xt)
        score = tf.reshape(score, tf.shape(x))
        noise = tf.random.normal(
            shape=tf.shape(x),
            mean=0.0,
            stddev=tf.sqrt(tf.cast(beta_lst[t], tf.float32)),
        )
        beta_t = beta_lst[t]
        x = (1 + 0.5 * beta_t) * x + beta_t * score + noise
        x_lst.append(x)
    return tf.stack(x_lst[::-1], axis=0)


score_model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(L**2 + 1,), name="input_layer"),  # (x, t)
        keras.layers.Dense(256, activation="tanh", name="dense_1"),
        keras.layers.Dense(64, activation="tanh", name="dense_2"),
        keras.layers.Dense(16, activation="tanh", name="dense_3"),
        keras.layers.Dense(64, activation="tanh", name="dense_4"),
        keras.layers.Dense(L**2, activation="linear", name="output_layer"),
        keras.layers.Reshape((L, L), name="reshape_layer"),
    ]
)
score_model.summary()

# 学習データ
n_samples = 1000
sample_input = dict(name="ising_2d_sample", L=L, kT=kT, J=J, h=h, n_samples=n_samples)
x_ini = load_np(sample_input)
if x_ini is None:
    init = np.ones((L, L), dtype=int)
    x_lst = pure_metropolis(init, beta=1.0 / kT, J=J, h=h, n_steps=n_samples * 5)
    x_ini = x_lst[-n_samples:]
    save_np(x_ini, sample_input)
x_ini = tf.convert_to_tensor(x_ini, dtype=tf.float32)

# 拡散過程のパラメータ
T = 100  # 拡散ステップ数
beta_lst = np.linspace(0.01, 0.1, T)  # 拡散係数のリスト
alpha_lst = 1 - beta_lst
alpha_cumprod = np.cumprod(alpha_lst)

optimized = keras.optimizers.Adam(learning_rate=0.001)

alpha_cumprod_tensor = tf.convert_to_tensor(alpha_cumprod, dtype=tf.float32)


@tf.function
def train_step(
    score_model: keras.Model, x_initial: tf.Tensor, t_values: tf.Tensor, T: int
) -> None:
    with tf.GradientTape() as tape:
        alpha_t = tf.gather(alpha_cumprod_tensor, t_values)
        noise = tf.random.normal(
            shape=tf.shape(x_initial),
            mean=0.0,
            stddev=tf.sqrt(1 - alpha_t)[:, None, None],
        )
        x_t = tf.sqrt(alpha_t)[:, None, None] * x_initial + noise
        x_t_flat = tf.reshape(x_t, (-1, L**2))
        t_norm = tf.divide(tf.cast(t_values, tf.float32), tf.cast(T, tf.float32))
        xt_input = tf.concat([x_t_flat, t_norm[:, None]], axis=1)
        predicted_score = score_model(xt_input)
        target_score = noise / tf.sqrt(1 - alpha_t)[:, None, None]
        loss = tf.reduce_mean(tf.square(predicted_score - target_score))

    gradients = tape.gradient(loss, score_model.trainable_variables)
    if gradients is not None:
        optimized.apply_gradients(zip(gradients, score_model.trainable_variables))


loaded_model = keras.models.load_model("models/score_model_ising_2d.keras")
if loaded_model is not None:
    retrain = (
        input("Score model already exists. Do you want to retrain? (y/n): ")
        .strip()
        .lower()
    )
else:
    retrain = "y"

if retrain == "y":
    for epoch in tqdm(range(5000), desc="Training"):
        idx = tf.random.uniform([n_samples], minval=0, maxval=T, dtype=tf.int32)
        train_step(score_model, x_initial=x_ini, t_values=idx, T=T)

    score_model.save("models/score_model_ising_2d.keras")
loaded_model = keras.models.load_model("models/score_model_ising_2d.keras")

x_fin = tf.reshape(x_ini, (-1, L**2))
x_asymp = tf.random.normal(shape=tf.shape(x_fin), mean=0.0, stddev=1.0)
x_reconstructed = diffusion_backward(x_asymp, beta_lst, loaded_model)  # type: ignore
print(x_reconstructed.shape)
# 結果の可視化
x_ini_np = np.array(x_ini)
x_reconstructed_np = np.array(x_reconstructed)[-1].reshape(-1, 20, 20)
x_reconstructed_np = np.where(x_reconstructed_np >= 0, 1, -1)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(x_ini_np[i * 100], cmap="gray", vmin=-1, vmax=1)
    axes[0, i].set_title(f"Initial Sample {i + 1}")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_reconstructed_np[i], cmap="gray", vmin=-1, vmax=1)
    axes[1, i].set_title(f"Reconstructed Sample {i + 1}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
