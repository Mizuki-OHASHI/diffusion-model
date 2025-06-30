# 2 次元イジングモデル (拡散モデル) - 実空間U-Net版

import japanize_matplotlib  # noqa: F401
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from b_pure_metoropolis_2d_ising import ising_energy, pure_metropolis
from onsager import onsager_energy

np.random.seed(42)
tf.random.set_seed(42)


# ===============================================================
# モデルアーキテクチャと訓練・生成関数
# ===============================================================


def create_unet_model(input_shape, base_filters=32):
    """
    U-Netアーキテクチャのスコアモデルを作成する関数
    """
    inputs_img = keras.layers.Input(shape=input_shape)
    inputs_time = keras.layers.Input(shape=(1,))

    h, w = input_shape[0], input_shape[1]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    padding_config = ((0, pad_h), (0, pad_w))

    x = keras.layers.ZeroPadding2D(padding=padding_config)(inputs_img)

    time_embedding = keras.layers.Dense(base_filters * 4)(inputs_time)
    time_embedding = keras.layers.Activation("swish")(time_embedding)
    time_embedding = keras.layers.Dense(base_filters * 4)(time_embedding)

    skips = []
    # Encoder
    for filters in [base_filters, base_filters * 2]:
        x = keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="swish"
        )(x)
        x = keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="swish"
        )(x)
        skips.append(x)
        x = keras.layers.AveragePooling2D(2)(x)

    # Bottleneck
    x = keras.layers.Conv2D(
        base_filters * 4, kernel_size=3, padding="same", activation="swish"
    )(x)
    time_add = keras.layers.Dense(base_filters * 4)(time_embedding)
    x = keras.layers.Add()([x, time_add[:, None, None, :]])
    x = keras.layers.Conv2D(
        base_filters * 4, kernel_size=3, padding="same", activation="swish"
    )(x)

    # Decoder
    for filters in [base_filters * 2, base_filters]:
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Concatenate()([x, skips.pop()])
        x = keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="swish"
        )(x)
        x = keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="swish"
        )(x)

    # <--- 変更点: 出力チャンネル数は常に1
    x = keras.layers.Conv2D(1, kernel_size=1, padding="same", activation="linear")(x)

    outputs = keras.layers.Cropping2D(cropping=padding_config)(x)
    model = keras.Model(inputs=[inputs_img, inputs_time], outputs=outputs)
    return model


def diffusion_backward(x_asym, beta_lst, score_model, T):
    """拡散過程の後退ステップを実行する関数"""
    x = x_asym
    x_lst = [x]
    for t in tqdm(range(T - 1, -1, -1), desc="Sampling"):
        # 時刻tを[0,1]に正規化し、U-Netの入力形状に合わせる
        t_norm = tf.ones_like(x[:, 0:1, 0:1, 0:1]) * t / T
        score = score_model([x, t_norm[:, 0, 0, :]])

        noise_std = tf.sqrt(beta_lst[t]) if beta_lst[t] > 0 else 0
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_std)
        beta_t = beta_lst[t]

        # DDPMの更新式（スコアバージョン）を使用するとより安定する可能性がある
        # x = (1 / tf.sqrt(1.0 - beta_t)) * (x - (beta_t / tf.sqrt(1.0 - alpha_cumprod[t])) * score) + noise
        x = (1 + 0.5 * beta_t) * x + beta_t * score + noise  # 元の更新式
        x_lst.append(x)
    return tf.stack(x_lst[::-1], axis=0)


@tf.function
def train_step(score_model, x_ini, t_values, T, optimizer, alpha_cumprod):
    """訓練の1ステップを実行する関数"""
    with tf.GradientTape() as tape:
        alpha_t = tf.gather(alpha_cumprod, t_values)

        # 正規分布ノイズを生成
        noise = tf.random.normal(shape=tf.shape(x_ini))

        # ノイズを付加 (x_tを計算)
        x_t = (
            tf.sqrt(alpha_t)[:, None, None, None] * x_ini
            + tf.sqrt(1 - alpha_t)[:, None, None, None] * noise
        )

        # 時刻tを[0,1]に正規化
        t_norm = tf.divide(tf.cast(t_values, tf.float32), tf.cast(T, tf.float32))

        # スコアを予測
        predicted_score = score_model([x_t, t_norm[:, None]])

        # ターゲットスコアは -noise/stddev なので、-noise/(sqrt(1-alpha_t))
        # 今回はノイズそのものを予測する形式（DDPMで一般的）に変更
        target = noise
        loss = tf.reduce_mean(tf.square(predicted_score - target))

    gradients = tape.gradient(loss, score_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, score_model.trainable_variables))
    return loss


# ===============================================================
# メイン処理
# ===============================================================

# --- パラメータ設定 ---
kT = 2.0  # 温度
beta = 1 / kT
J = 1.0  # 相互作用定数
h = 0.0  # 外部磁場
epochs = 1000
n_samples = 5000
L = 20
batch_size = 64

# --- 訓練データ生成 ---
print("Generating training data with Metropolis...")
x_ini = np.ones((L, L))
# x_samplesの形状は (n_samples, L, L)

x_samples = pure_metropolis(
    x_ini,
    beta=beta,
    J=J,
    h=h,
    n_mcs_equilibration=1000,
    n_mcs_measurement=n_samples,
    measurement_interval_mcs=10,
)

# --- データ前処理 (実空間U-Net用) ---
# (n_samples, L, L) -> (n_samples, L, L, 1)
x_lst = x_samples[2][..., np.newaxis]
x_lst = tf.convert_to_tensor(x_lst, dtype=tf.float32)
print(f"Data shape for Real-Space U-Net: {x_lst.shape}")

# --- 拡散プロセス設定 ---
T = 1000
noise_beta = np.linspace(1e-4, 0.02, T)
alpha_lst = 1 - noise_beta
alpha_cumprod = np.cumprod(alpha_lst)
beta_tensor = tf.convert_to_tensor(noise_beta, dtype=tf.float32)
alpha_cumprod_tensor = tf.convert_to_tensor(alpha_cumprod, dtype=tf.float32)

# --- モデル定義 (U-Net) ---
unet_input_shape = x_lst.shape[1:]  # (L, L, 1)
score_model = create_unet_model(unet_input_shape)
score_model.summary()
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

# --- 訓練ループ ---
# train_stepの引数から不要なbeta_tensorを削除
for epoch in tqdm(range(epochs), desc="Training"):
    data_indices = tf.random.uniform(
        [batch_size], minval=0, maxval=x_lst.shape[0], dtype=tf.int32
    )
    x_0_batch = tf.gather(x_lst, data_indices)
    t_batch = tf.random.uniform([batch_size], minval=0, maxval=T, dtype=tf.int32)
    loss = train_step(
        score_model, x_0_batch, t_batch, T, optimizer, alpha_cumprod_tensor
    )
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

# --- モデルの保存と読み込み ---
print("Saving model...")
score_model.save("models/b_diffusion_model_2d_ising_realspace_unet.keras")
print("Loading model...")
score_model = keras.models.load_model(
    "models/b_diffusion_model_2d_ising_realspace_unet.keras"
)

# --- サンプリング（生成） ---
samples = 1000
x_asym = tf.random.normal(shape=(samples, *unet_input_shape), mean=0.0, stddev=1.0)
x_rec = diffusion_backward(x_asym, beta_tensor, score_model, T)

# --- 後処理と評価 ---
x_rec_np = np.array(x_rec)[0]

x_rec_np = np.where(x_rec_np > 0, 1, -1)
# ising_energyやimshowのためにチャンネル次元を削除 (samples, L, L, 1) -> (samples, L, L)
x_rec_np_final = np.squeeze(x_rec_np, axis=-1)

# 訓練データは元のx_samplesをそのまま使える
x_lst_np = x_samples

onsager_e = onsager_energy(kT, J)
data_energy = np.mean(ising_energy(x_lst_np, J, h) / L**2)
rec_energy = np.mean(ising_energy(x_rec_np_final, J, h) / L**2)
print(f"Onsager Energy: {onsager_e:.4f}")
print(f"Data Energy: {data_energy:.4f}")
print(f"Reconstructed Energy: {rec_energy:.4f}")

# --- 可視化 ---
fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)
fig.suptitle(
    f"kT={kT}, Onsager={onsager_e:.3f}, Data={data_energy:.3f}, Generated={rec_energy:.3f}",
    fontsize=16,
)
for i in range(5):
    random_idx = np.random.randint(0, x_lst_np.shape[0])
    axes[0, i].imshow(x_lst_np[random_idx], cmap="gray", vmin=-1, vmax=1)
    axes[0, i].set_title("Input Data")
    axes[0, i].axis("off")

    axes[1, i].imshow(x_rec_np_final[i], cmap="gray", vmin=-1, vmax=1)
    axes[1, i].set_title("Generated Sample")
    axes[1, i].axis("off")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
