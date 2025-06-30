# 2 次元イジングモデル (拡散モデル)

import japanize_matplotlib  # noqa: F401
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from b_fft import ising_fft, ising_ifft
from b_pure_metoropolis_2d_ising import ising_energy, pure_metropolis
from onsager import onsager_energy
from utils import load_np, save_np

np.random.seed(42)
tf.random.set_seed(42)


def diffusion_backward(x_asym, beta_lst, score_model, T):
    """
    拡散過程の後退ステップを実行する関数
    """
    x = x_asym
    x_lst = [x]
    for t in range(T - 1, -1, -1):
        t_norm = tf.ones_like(x[:, 0:1]) * t / T  # (samples, 1)

        xt = tf.concat([x, t_norm], axis=1)  # (samples, L^2 + 1)

        score = score_model(xt)
        noise = tf.random.normal(
            shape=tf.shape(x), mean=0.0, stddev=tf.sqrt(beta_lst[t])
        )
        beta_t = beta_lst[t]

        x = (1 + 0.5 * beta_t) * x + beta_t * score + noise
        x_lst.append(x)
    return tf.stack(x_lst[::-1], axis=0)


@tf.function
def train_step(score_model, x_ini, t_values, T, optimizer, beta_lst, alpha_cumprod):
    with tf.GradientTape() as tape:
        # 摂動カーネルのノイズスケジュール
        # sigma_t = tf.sqrt(tf.gather(beta_tensor, t_values))
        alpha_t = tf.gather(alpha_cumprod, t_values)

        # ノイズを加える
        noise = tf.random.normal(
            shape=tf.shape(x_ini), mean=0.0, stddev=tf.sqrt(1 - alpha_t)[:, None]
        )
        x_t = tf.sqrt(alpha_t[:, None]) * x_ini + noise

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
    else:
        raise ValueError("Gradients are None, check the model and inputs.")


kT = 2.0  # 温度
beta = 1 / kT
J = 1.0  # 相互作用定数
h = 0.0  # 外部磁場
epochs = 10000
n_samples = 50000
L = 20
x_ini = np.ones((L, L))
x_samples = pure_metropolis(
    x_ini,
    beta=beta,
    J=J,
    h=h,
    n_mcs_equilibration=10000,
    n_mcs_measurement=n_samples,
    measurement_interval_mcs=10,
)
x_lst = ising_fft(x_samples, L)
x_lst = np.concat([x_lst.real, x_lst.imag], axis=-1)
x_lst = x_lst.reshape(-1, 2 * (L // 2 + 1) * L)
x_lst = tf.convert_to_tensor(x_lst, dtype=tf.float32)

noise_beta = np.linspace(1e-4, 0.02, 1000)
alpha_lst = 1 - noise_beta
alpha_cumprod = np.cumprod(alpha_lst)
T = noise_beta.shape[0]
beta_tensor = tf.convert_to_tensor(noise_beta, dtype=tf.float32)
alpha_cumprod_tensor = tf.convert_to_tensor(alpha_cumprod, dtype=tf.float32)

score_model = keras.Sequential(
    [
        keras.layers.Input(shape=(2 * (L // 2 + 1) * L + 1,)),  # type: ignore
        keras.layers.Dense(2 * (L // 2 + 1) * L, activation="tanh"),
        keras.layers.Dense(2 * (L // 2 + 1) * L, activation="tanh"),
        keras.layers.Dense(2 * (L // 2 + 1) * L, activation="tanh"),
        keras.layers.Dense(2 * (L // 2 + 1) * L, activation="tanh"),
        keras.layers.Dense(2 * (L // 2 + 1) * L, activation="linear"),
    ]
)
score_model.summary()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

batch_size = 64

for epoch in tqdm(range(epochs), desc="Training"):
    data_indices = tf.random.uniform(
        [batch_size], minval=0, maxval=x_lst.shape[0], dtype=tf.int32
    )
    x_0_batch = tf.gather(x_lst, data_indices)
    t_batch = tf.random.uniform([batch_size], minval=0, maxval=T, dtype=tf.int32)

    train_step(
        score_model, x_0_batch, t_batch, T, optimizer, beta_tensor, alpha_cumprod_tensor
    )


# モデルの保存
score_model.save("models/b_diffusion_model_2d_ising.keras")

# モデルの読み込み
score_model = keras.models.load_model("models/b_diffusion_model_2d_ising.keras")

samples = 5
x_asym = tf.random.normal(shape=(samples, 2 * (L // 2 + 1) * L), mean=0.0, stddev=1.0)
x_rec = diffusion_backward(x_asym, beta_tensor, score_model, T)

x_rec_np = np.array(x_rec)[0].reshape(samples, L, -1)
x_rec_np_r = x_rec_np[..., : L // 2 + 1]
x_rec_np_i = x_rec_np[..., L // 2 + 1 :]
x_rec_np = ising_ifft(x_rec_np_r + 1j * x_rec_np_i, L)
x_rec_np = np.where(x_rec_np > 0, 1, -1)  # 二値化

x_lst_np = np.array(x_lst).reshape(n_samples, L, -1)
x_lst_np_r = x_lst_np[..., : L // 2 + 1]
x_lst_np_i = x_lst_np[..., L // 2 + 1 :]
x_lst_np = ising_ifft(x_lst_np_r + 1j * x_lst_np_i, L)
x_lst_np = np.where(x_lst_np > 0, 1, -1)  # 二値化

onsager_energy = onsager_energy(kT, J)
data_energy = np.mean(ising_energy(x_lst_np, J, h) / L**2)
rec_energy = np.mean(ising_energy(x_rec_np, J, h) / L**2)
print(f"Onsager Energy: {onsager_energy:.4f}")
print(f"Data Energy: {data_energy:.4f}")
print(f"Reconstructed Energy: {rec_energy:.4f}")


fig, axes = plt.subplots(2, 5, figsize=(10, 3), sharex=True, sharey=True)
for i in range(5):
    axes[0, i].imshow(
        x_lst_np[i * 100],
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    axes[1, i].imshow(
        x_rec_np[i].reshape(L, L),
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    axes[0, i].set_title(f"Input {i + 1}")
    axes[0, i].axis("off")
    axes[1, i].set_title(f"Sample {i + 1}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
