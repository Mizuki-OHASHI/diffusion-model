# 2 次元イジングモデル (拡散モデル) - U-Net修正版

import math

import japanize_matplotlib  # noqa: F401
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from b_fft import ising_fft, ising_ifft
from b_pure_metoropolis_2d_ising import ising_energy, pure_metropolis
from onsager import onsager_energy

np.random.seed(42)
tf.random.set_seed(42)


# ===============================================================
# モデルアーキテクチャと訓練・生成関数
# ===============================================================


def create_unet_model(
    input_shape, base_filters=32
):  # <--- 変更点: U-Netモデル作成関数を追加
    """
    U-Netアーキテクチャのスコアモデルを作成する関数
    """
    inputs_img = keras.layers.Input(shape=input_shape)
    inputs_time = keras.layers.Input(shape=(1,))

    # --- パディングとクロッピングの計算 ---
    # 2回プーリングするため、高さと幅が4の倍数になるように調整
    h, w = input_shape[0], input_shape[1]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    # 右と下側にパディングを追加
    padding_config = ((0, pad_h), (0, pad_w))

    # --- モデル本体 ---
    # 入力パディング
    x = keras.layers.ZeroPadding2D(padding=padding_config)(inputs_img)

    time_embedding = keras.layers.Dense(base_filters * 4)(inputs_time)
    time_embedding = keras.layers.Activation("swish")(time_embedding)
    time_embedding = keras.layers.Dense(base_filters * 4)(time_embedding)

    skips = []

    # Encoder
    for filters in [base_filters, base_filters * 2]:
        x = keras.layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("swish")(x)
        skips.append(x)
        x = keras.layers.AveragePooling2D(2)(x)

    # Bottleneck
    x = keras.layers.Conv2D(base_filters * 4, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("swish")(x)
    time_add = keras.layers.Dense(base_filters * 4)(time_embedding)
    x = keras.layers.Add()([x, time_add[:, None, None, :]])
    x = keras.layers.Conv2D(base_filters * 4, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("swish")(x)

    # Decoder
    for filters in [base_filters * 2, base_filters]:
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Concatenate()([x, skips.pop()])
        x = keras.layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("swish")(x)

    x = keras.layers.Conv2D(
        input_shape[-1], kernel_size=1, padding="same", activation="linear"
    )(x)

    # 出力クロッピング
    outputs = keras.layers.Cropping2D(cropping=padding_config)(x)

    model = keras.Model(inputs=[inputs_img, inputs_time], outputs=outputs)
    return model


def diffusion_backward(x_asym, beta_lst, score_model, T):
    """
    拡散過程の後退ステップを実行する関数
    """
    x = x_asym
    x_lst = [x]
    for t in tqdm(range(T - 1, -1, -1), desc="Sampling"):
        t_norm = tf.ones_like(x[:, 0:1, 0:1, 0:1]) * t / T

        # <--- 変更点: U-Netモデルへの入力形式を修正
        score = score_model([x, t_norm[:, 0, 0, :]])  # t_normは(samples, 1)の形にする

        noise_std = tf.sqrt(beta_lst[t]) if beta_lst[t] > 0 else 0
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_std)
        beta_t = beta_lst[t]

        x = (1 + 0.5 * beta_t) * x + beta_t * score + noise
        x_lst.append(x)
    return tf.stack(x_lst[::-1], axis=0)


@tf.function
def train_step(score_model, x_ini, t_values, T, optimizer, beta_lst, alpha_cumprod):
    with tf.GradientTape() as tape:
        alpha_t = tf.gather(alpha_cumprod, t_values)
        noise_std = tf.sqrt(1 - alpha_t)

        noise = tf.random.normal(shape=tf.shape(x_ini))
        # ブロードキャストのために形状を調整
        x_t = (
            tf.sqrt(alpha_t)[:, None, None, None] * x_ini
            + noise_std[:, None, None, None] * noise
        )

        t_norm = tf.divide(tf.cast(t_values, tf.float32), tf.cast(T, tf.float32))

        # <--- 変更点: U-Netモデルへの入力形式を修正
        predicted_score = score_model([x_t, t_norm[:, None]])

        target_score = -noise
        loss = tf.reduce_mean(tf.square(predicted_score - target_score))

    gradients = tape.gradient(loss, score_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, score_model.trainable_variables))
    return loss


# ===============================================================
# メイン処理
# ===============================================================

# --- パラメータ設定 ---
kT = 2.0
beta = 1 / kT
J = 1.0
h = 0.0
n_steps = 100000
epochs = 1000
n_samples = n_steps // 2
L = 20
batch_size = 64

# --- 訓練データ生成 ---
print("Generating training data with Metropolis...")
x_ini = np.ones((L, L))
x_samples = pure_metropolis(x_ini, beta=beta, J=J, h=h, n_steps=n_steps)[-n_samples:]

# --- データ前処理 (U-Net用) ---
x_lst = ising_fft(x_samples, L)
# <--- 変更点: データを(H, W, C)形式に整形
x_lst = np.stack([x_lst.real, x_lst.imag], axis=-1)
x_lst = tf.convert_to_tensor(x_lst, dtype=tf.float32)
print(f"Data shape for U-Net: {x_lst.shape}")


# --- 拡散プロセス設定 ---
T = 1000  # <--- 改善点: タイムステップ数を増やす
noise_beta = np.linspace(1e-4, 0.02, T)  # <--- 改善点: より一般的なスケジュールに変更
alpha_lst = 1 - noise_beta
alpha_cumprod = np.cumprod(alpha_lst)
beta_tensor = tf.convert_to_tensor(noise_beta, dtype=tf.float32)
alpha_cumprod_tensor = tf.convert_to_tensor(alpha_cumprod, dtype=tf.float32)

# --- モデル定義 (U-Net) ---
# <--- 変更点: U-Netモデルをインスタンス化
unet_input_shape = x_lst.shape[1:]
score_model = create_unet_model(unet_input_shape)
score_model.summary()
optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # <--- 改善点: 学習率を少し下げる

# --- 訓練ループ ---
for epoch in tqdm(range(epochs), desc="Training"):
    data_indices = tf.random.uniform(
        [batch_size], minval=0, maxval=x_lst.shape[0], dtype=tf.int32
    )
    x_0_batch = tf.gather(x_lst, data_indices)
    t_batch = tf.random.uniform([batch_size], minval=0, maxval=T, dtype=tf.int32)
    loss = train_step(
        score_model, x_0_batch, t_batch, T, optimizer, beta_tensor, alpha_cumprod_tensor
    )
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")


# --- モデルの保存と読み込み ---
print("Saving model...")
score_model.save("models/b_diffusion_model_2d_ising_unet.keras")
print("Loading model...")
score_model = keras.models.load_model("models/b_diffusion_model_2d_ising_unet.keras")

# --- サンプリング（生成） ---
samples = 5
x_asym = tf.random.normal(shape=(samples, *unet_input_shape), mean=0.0, stddev=1.0)
x_rec = diffusion_backward(x_asym, beta_tensor, score_model, T)

# --- 後処理と評価 ---
x_rec_np = np.array(x_rec)[0]  # 最終ステップ(t=0)の結果を取得

# <--- 変更点: (H,W,C)形式からのデータ展開
x_rec_np_r = x_rec_np[..., 0]
x_rec_np_i = x_rec_np[..., 1]
x_rec_np = ising_ifft(x_rec_np_r + 1j * x_rec_np_i, L)
x_rec_np = np.where(x_rec_np > 0, 1, -1)

# 訓練データも同様に変換して比較
x_lst_np = np.array(x_lst)
# <--- 変更点: (H,W,C)形式からのデータ展開
x_lst_np_r = x_lst_np[..., 0]
x_lst_np_i = x_lst_np[..., 1]
x_lst_np = ising_ifft(x_lst_np_r + 1j * x_lst_np_i, L)
x_lst_np = np.where(x_lst_np > 0, 1, -1)

onsager_e = onsager_energy(kT, J)
data_energy = np.mean(ising_energy(x_lst_np, J, h) / L**2)
rec_energy = np.mean(ising_energy(x_rec_np, J, h) / L**2)
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
    # 訓練データをランダムに選んで表示
    random_idx = np.random.randint(0, x_lst_np.shape[0])
    axes[0, i].imshow(x_lst_np[random_idx], cmap="gray", vmin=-1, vmax=1)
    axes[0, i].set_title(f"Input Data")
    axes[0, i].axis("off")

    axes[1, i].imshow(x_rec_np[i].reshape(L, L), cmap="gray", vmin=-1, vmax=1)
    axes[1, i].set_title(f"Generated Sample")
    axes[1, i].axis("off")

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
plt.show()
