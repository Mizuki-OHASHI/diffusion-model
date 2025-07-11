# -*- coding: utf-8 -*-
"""MNIST-diffusion.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qn6dGwCoXt4iGl6pX5oGNtwFYRbbg8_p

# MNIST diffusion

## Setup
"""

import os
import datetime
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
from google.colab import drive

np.random.seed(42)
tf.random.set_seed(42)

drive.mount("/content/drive")
os.chdir("/content/drive/MyDrive/Colab Notebooks/models")

gpu_devices = tf.config.list_physical_devices("GPU")
print(f"Num GPUs Available: {len(gpu_devices)}")

if gpu_devices:
    print("GPU devices:")
    for device in gpu_devices:
        print(f"  {device}")

(X, Y), (_, _) = keras.datasets.mnist.load_data()
X_b = np.where(X < 64, 0, 1)
plt.imshow(X_b[0], cmap="gray")

"""## Models"""

def diffusion_backward(X_b, model, time_step):
    X = X_b
    X_lst = [X]
    for t in range(time_step-1, -1, -1):
        t_tensor = tf.cast(tf.ones(shape=X_b.shape[0]) * t / time_step, dtype=tf.float32)
        flip_proba = model([X, t_tensor])
        flip = tf.random.uniform(shape=X_b.shape) < flip_proba
        X = X ^ flip
        X_lst.append(X)
    return tf.stack(X_lst)

# def diffusion_backward(X_b, model, time_step, flip_proba):
#     X = X_b
#     X_lst = [X]
#     for t in range(time_step-1, -1, -1):
#         t_tensor = tf.cast(tf.ones(shape=X_b.shape[0]) * t / time_step, dtype=tf.float32)
#         output = model([X, t_tensor]) # batch of (X_0, flip)
#         X = output[..., 0]
#         # flip = tf.random.uniform(shape=X_b.shape) < tf.gather(flip_proba, t)
#         # X = X_b ^ flip
#         X_lst.append(X)
#     return tf.stack(X_lst)

# 拡散ステップ t をベクトルに変換するためのカスタムレイヤー
class SinusoidalPositionalEmbedding(layers.Layer):
    """
    制限波による位置埋め込みレイヤー
    """

    def __init__(self, dim, max_positions=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_positions = max_positions

    def call(self, inputs):
        # inputs は (batch_size, 1) のタイムステップ
        positions = tf.cast(inputs, tf.float32)
        half_dim = self.dim // 2
        freqs = tf.math.exp(
            -tf.math.log(float(self.max_positions)) * tf.range(0, half_dim, dtype=tf.float32) / (half_dim - 1)
        )
        # (1, dim/2)
        args = positions[:, tf.newaxis] * freqs[tf.newaxis, :]
        # (batch_size, dim/2)
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], axis=-1)
        # (batch_size, dim)
        if self.dim % 2 != 0:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_positions": self.max_positions,
        })
        return config

def create_diffusion_unet(img_size=28, time_emb_dim=32):
    """
    シンプルなU-Netモデル
    """
    # === 入力層 ===
    image_input = layers.Input(shape=(img_size, img_size, 1), name="image_input")
    # タイムステップは整数として入力
    time_input = layers.Input(shape=(), dtype=tf.int64, name="time_input")

    # === 時間埋め込み ===
    # SinusoidalPositionalEmbeddingで時間情報をベクトル化
    time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)(time_input)
    # MLPを通して時間情報をさらに加工
    time_embedding = layers.Dense(time_emb_dim * 4, activation="swish")(time_embedding)
    time_embedding = layers.Dense(time_emb_dim * 4, activation="swish")(time_embedding)

    # === エンコーダー ===
    # 28x28 -> 14x14
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="swish")(image_input)
    # スキップ接続用に特徴マップを保存 (28x28x4)
    skip1 = layers.Conv2D(4, kernel_size=3, padding="same", activation="swish")(x)
    x = layers.MaxPooling2D(2)(skip1)

    # 14x14 -> 7x7
    x = layers.Conv2D(8, kernel_size=3, padding="same", activation="swish")(x)
    # スキップ接続用に特徴マップを保存 (14x14x8)
    skip2 = layers.Conv2D(8, kernel_size=3, padding="same", activation="swish")(x)
    x = layers.MaxPooling2D(2)(skip2)

    # === ボトルネック ===
    x = layers.Conv2D(8, kernel_size=3, padding="same", activation="swish")(x)

    # 時間埋め込みを注入
    time_emb_mlp = layers.Dense(8, activation="swish")(time_embedding) # チャネル数に合わせる
    # (batch, 1, 1, 8) にリシェイプして加算できるようにする
    time_emb_mlp = layers.Reshape((1, 1, 8))(time_emb_mlp)
    x = layers.Add()([x, time_emb_mlp])

    x = layers.Conv2D(8, kernel_size=3, padding="same", activation="swish")(x)


    # === デコーダー ===
    # 7x7 -> 14x14
    x = layers.UpSampling2D(2)(x)
    # スキップ接続を結合 (concat)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(4, kernel_size=3, padding="same", activation="swish")(x)

    # 14x14 -> 28x28
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(4, kernel_size=3, padding="same", activation="swish")(x)

    # === 出力層 ===
    # 最終的な出力チャネル数を1に調整
    outputs = layers.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    # モデルの定義
    model = models.Model([image_input, time_input], outputs, name="simple_diffusion_unet")
    return model

def create_improved_diffusion_unet(img_size=28, time_emb_dim=32):
    """
    残差ブロック、GroupNormalization、複数回の時間埋め込み注入を導入した
    改善版のU-Netモデル
    """
    # === ヘルパー関数: 残差ブロック ===
    def ResBlock(width):
        def apply(x, time_emb):
            input_width = x.shape[-1]

            # 時間埋め込みを加工して注入
            time_mlp = layers.Dense(width, activation="swish")(time_emb)
            time_mlp = layers.Reshape((1, 1, width))(time_mlp)

            # メインのパス
            h = layers.GroupNormalization(groups=8)(x)
            h = layers.Activation("swish")(h)
            h = layers.Conv2D(width, kernel_size=3, padding="same")(h)

            h = layers.Add()([h, time_mlp]) # 時間情報を加算

            h = layers.GroupNormalization(groups=8)(h)
            h = layers.Activation("swish")(h)
            h = layers.Conv2D(width, kernel_size=3, padding="same")(h)

            # 残差接続
            if input_width != width:
                # チャンネル数が異なる場合は1x1畳み込みで合わせる
                x = layers.Conv2D(width, kernel_size=1, padding="same")(x)

            return layers.Add()([x, h])
        return apply

    # === 入力層 ===
    image_input = layers.Input(shape=(img_size, img_size, 1), name="image_input")
    time_input = layers.Input(shape=(), dtype=tf.int64, name="time_input")

    # === 時間埋め込み ===
    time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)(time_input)
    # MLPを通して時間情報をさらに加工
    time_embedding = layers.Dense(time_emb_dim * 4, activation="swish")(time_embedding)
    time_embedding = layers.Dense(time_emb_dim * 4, activation="swish")(time_embedding)

    # === エンコーダー ===
    # 初期畳み込み
    x = layers.Conv2D(32, kernel_size=1, padding="same")(image_input)

    # 28x28
    x = ResBlock(width=32)(x, time_embedding)
    skip1 = x
    x = layers.MaxPooling2D(pool_size=2)(x)

    # 14x14
    x = ResBlock(width=64)(x, time_embedding)
    skip2 = x
    x = layers.MaxPooling2D(pool_size=2)(x)

    # === ボトルネック ===
    # 7x7
    x = ResBlock(width=128)(x, time_embedding)

    # === デコーダー ===
    # 7x7 -> 14x14
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Concatenate()([x, skip2]) # スキップ接続を結合
    x = ResBlock(width=64)(x, time_embedding)

    # 14x14 -> 28x28
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Concatenate()([x, skip1]) # スキップ接続を結合
    x = ResBlock(width=32)(x, time_embedding)

    # === 出力層 ===
    # チャネル数を1に調整
    outputs = layers.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    # モデルの定義
    model = models.Model([image_input, time_input], outputs, name="improved_diffusion_unet")
    return model

def create_2channel_diffusion_unet(img_size=28, time_emb_dim=32):
    """
    残差ブロック、GroupNormalization、複数回の時間埋め込み注入を導入した
    改善版のU-Netモデル
    出力にノイズ前の画像を追加
    """
    # === ヘルパー関数: 残差ブロック ===
    def ResBlock(width):
        def apply(x, time_emb):
            input_width = x.shape[-1]

            # 時間埋め込みを加工して注入
            time_mlp = layers.Dense(width, activation="swish")(time_emb)
            time_mlp = layers.Reshape((1, 1, width))(time_mlp)

            # メインのパス
            h = layers.GroupNormalization(groups=8)(x)
            h = layers.Activation("swish")(h)
            h = layers.Conv2D(width, kernel_size=3, padding="same")(h)

            h = layers.Add()([h, time_mlp]) # 時間情報を加算

            h = layers.GroupNormalization(groups=8)(h)
            h = layers.Activation("swish")(h)
            h = layers.Conv2D(width, kernel_size=3, padding="same")(h)

            # 残差接続
            if input_width != width:
                # チャンネル数が異なる場合は1x1畳み込みで合わせる
                x = layers.Conv2D(width, kernel_size=1, padding="same")(x)

            return layers.Add()([x, h])
        return apply

    # === 入力層 ===
    image_input = layers.Input(shape=(img_size, img_size, 1), name="image_input")
    time_input = layers.Input(shape=(), dtype=tf.int64, name="time_input")

    # === 時間埋め込み ===
    time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)(time_input)
    # MLPを通して時間情報をさらに加工
    time_embedding = layers.Dense(time_emb_dim * 4, activation="swish")(time_embedding)
    time_embedding = layers.Dense(time_emb_dim * 4, activation="swish")(time_embedding)

    # === エンコーダー ===
    # 初期畳み込み
    x = layers.Conv2D(32, kernel_size=1, padding="same")(image_input)

    # 28x28
    x = ResBlock(width=32)(x, time_embedding)
    skip1 = x
    x = layers.MaxPooling2D(pool_size=2)(x)

    # 14x14
    x = ResBlock(width=64)(x, time_embedding)
    skip2 = x
    x = layers.MaxPooling2D(pool_size=2)(x)

    # === ボトルネック ===
    # 7x7
    x = ResBlock(width=128)(x, time_embedding)

    # === デコーダー ===
    # 7x7 -> 14x14
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Concatenate()([x, skip2]) # スキップ接続を結合
    x = ResBlock(width=64)(x, time_embedding)

    # 14x14 -> 28x28
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Concatenate()([x, skip1]) # スキップ接続を結合
    x = ResBlock(width=32)(x, time_embedding)

    # === 出力層 ===
    # チャネル数を1に調整
    outputs = layers.Conv2D(2, kernel_size=1, padding="same", activation="sigmoid")(x)

    # モデルの定義
    model = models.Model([image_input, time_input], outputs, name="improved_diffusion_unet")
    return model

"""## Training"""

n_samples = 1000
X_5 = X_b[Y == 5][:n_samples]
X_5 = tf.convert_to_tensor(X_5, dtype=tf.bool)
X_5 = tf.expand_dims(X_5, axis=-1)
print("X_5.shape:", X_5.shape)

time_step = 100
forward_flip_proba = tf.linspace(0.005, 0.02, time_step)
forward_unflip_proba = 1 - forward_flip_proba
forward_unflip_cumprod = tf.math.cumprod(forward_unflip_proba)
forward_flip_cumprod = 1 - forward_unflip_cumprod

# model = create_2channel_diffusion_unet()
model = create_improved_diffusion_unet()
# model = create_diffusion_unet()
# model.summary()

@tf.function
def train_step(X_batch, model, optimizer, time_step, flip_cumprod, l1, l2):
    with tf.GradientTape() as tape:
        time_idx = tf.random.uniform(shape=(X_batch.shape[0],), minval=1, maxval=time_step, dtype=tf.int32)
        time_nrmlzd = tf.cast(time_idx / time_step, dtype=tf.float32)
        flip_cumprod_t = tf.gather(flip_cumprod, time_idx)
        flip = tf.random.uniform(shape=X_batch.shape) < flip_cumprod_t[:, None, None, None]
        X_fliped = X_batch ^ flip
        flip_proba_predicted = model([X_fliped, time_nrmlzd])

        # l1: 0に近づける正則化
        # l2: 0or1に近づける正則化
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.cast(flip, tf.float32), flip_proba_predicted)
        )

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

n_epochs = 5000
learning_rate = 0.001
batch_size = 128
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(n_epochs):
    batch_idx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=n_samples, dtype=tf.int32)
    X_batch = tf.gather(X_5, batch_idx)
    loss = train_step(X_batch, model, optimizer, time_step, forward_flip_cumprod, l1=0.01, l2=0.05)
    if epoch % (n_epochs // 10) == 0:
        print(f"\nEpoch {epoch+1:5d}, Loss: {loss:.6f}", end="")
    else:
        print(f"\rEpoch {epoch+1:5d}, Loss: {loss:.6f}", end="")
print()

"""<details><summary>mnist_diffusion_unet_20250701_195004</summary>

    Epoch  250, Loss: 0.104810
    Epoch  500, Loss: 0.082589
    Epoch  750, Loss: 0.078260
    Epoch 1000, Loss: 0.078841
    Epoch 1250, Loss: 0.076038
    Epoch 1500, Loss: 0.073197
    Epoch 1750, Loss: 0.069026
    Epoch 2000, Loss: 0.063473
    Epoch 2250, Loss: 0.061776
    Epoch 2500, Loss: 0.062990
</details>

<details><summary>mnist_diffusion_unet_20250701_195705</summary>

    Epoch 2750, Loss: 0.056665
    Epoch 3000, Loss: 0.061805
    Epoch 3250, Loss: 0.067896
    Epoch 3500, Loss: 0.058838
    Epoch 3750, Loss: 0.061784
    Epoch 4000, Loss: 0.058690
    Epoch 4250, Loss: 0.063214
    Epoch 4500, Loss: 0.056842
    Epoch 4750, Loss: 0.056462
    Epoch 5000, Loss: 0.058490
</details>

<details><summary>mnist_diffusion_unet_20250701_200727</summary>

    Epoch 5250, Loss: 0.054094
    Epoch 5500, Loss: 0.056725
    Epoch 5750, Loss: 0.052291
    Epoch 6000, Loss: 0.052407
    Epoch 6250, Loss: 0.050918
    Epoch 6500, Loss: 0.048564
    Epoch 6750, Loss: 0.055017
    Epoch 7000, Loss: 0.053196
    Epoch 7250, Loss: 0.052590
    Epoch 7500, Loss: 0.051980
    Epoch 7750, Loss: 0.049998
    Epoch 8000, Loss: 0.050555
    Epoch 8250, Loss: 0.052300
    Epoch 8500, Loss: 0.053661
    Epoch 8750, Loss: 0.054109
    Epoch 9000, Loss: 0.048477
    Epoch 9250, Loss: 0.054802
    Epoch 9500, Loss: 0.051236
    Epoch 9750, Loss: 0.049614
    Epoch 10000, Loss: 0.055239
</details>

出力層の活性化関数を sigmoid に変更

<details><summary>mnist_diffusion_unet_20250702_062904</summary>

    Epoch  500, Loss: 0.327876
    Epoch 1000, Loss: 0.086485
    Epoch 1500, Loss: 0.081932
    Epoch 2000, Loss: 0.079103
    Epoch 2500, Loss: 0.078150
    Epoch 3000, Loss: 0.075059
    Epoch 3500, Loss: 0.042110
    Epoch 4000, Loss: 0.045527
    Epoch 4500, Loss: 0.042607
    Epoch 5000, Loss: 0.044798
</details>

正則化 `1=0.01, l2=0.05`

<details><summary>mnist_diffusion_unet_20250702_072242</summary>

    Epoch   500, Loss: 0.049719
    Epoch  1000, Loss: 0.051376
    Epoch  1500, Loss: 0.051397
    Epoch  2000, Loss: 0.048221
    Epoch  2500, Loss: 0.049300
    Epoch  3000, Loss: 0.048261
    Epoch  3500, Loss: 0.048193
    Epoch  4000, Loss: 0.048056
    Epoch  4500, Loss: 0.050160
    Epoch  5000, Loss: 0.047777
</details>

バイナリクロスエントロピー

<details><summary>mnist_diffusion_unet_20250702_074922</summary>

    Epoch   500, Loss: 0.141760
    Epoch  1000, Loss: 0.148688
    Epoch  1500, Loss: 0.144939
    Epoch  2000, Loss: 0.140419
    Epoch  2500, Loss: 0.140758
    Epoch  3000, Loss: 0.133324
    Epoch  3500, Loss: 0.135422
    Epoch  4000, Loss: 0.136669
    Epoch  4500, Loss: 0.136507
    Epoch  5000, Loss: 0.135597
</details>

出力を2チャンネル (元画像, 反転) に増やした
(Cf. https://arxiv.org/abs/2501.13915 )

<details><summary>mnist_diffusion_unet_20250702_140010</summary>

    Epoch   500, Loss: 0.287811
    Epoch  1000, Loss: 0.291328
    Epoch  1500, Loss: 0.286344
    Epoch  2000, Loss: 0.274445
    Epoch  2500, Loss: 0.268278
    Epoch  3000, Loss: 0.261921
    Epoch  3500, Loss: 0.254842
    Epoch  4000, Loss: 0.280577
    Epoch  4500, Loss: 0.264389
    Epoch  5000, Loss: 0.278277
</details>

実は上のコードは全て逆拡散プロセスが間違っていた...

<details><summary>mnist_diffusion_unet_20250702_141510</summary>

    Epoch   500, Loss: 0.145560
    Epoch  1000, Loss: 0.145385
    Epoch  1500, Loss: 0.150824
    Epoch  2000, Loss: 0.137519
    Epoch  2500, Loss: 0.137336
    Epoch  3000, Loss: 0.129303
    Epoch  3500, Loss: 0.131737
    Epoch  4000, Loss: 0.132240
    Epoch  4500, Loss: 0.139037
    Epoch  5000, Loss: 0.129388
</details>

"""

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"mnist_diffusion_unet_{timestamp}.keras"
model.save(model_filename)
print(f"Model saved as {model_filename}")

n_generate = 10
X_asymp = np.random.randint(2, size=(n_generate, 28, 28, 1))
X_asymp = tf.convert_to_tensor(X_asymp, dtype=tf.bool)
X_lst = diffusion_backward(X_asymp, model, time_step=time_step)

# 最後の10枚の画像を生成して表示
fig, axes = plt.subplots(1, n_generate, figsize=(20, 2))
# 学習済みモデルを使ってノイズを除去
generated_images = X_lst[-1]

for i in range(n_generate):
    axes[i].imshow(generated_images[i, :, :, 0], cmap="gray")
    axes[i].axis("off")
plt.suptitle("Generated Images")
figname = model_filename.replace(".keras", ".png")
plt.savefig(f"../figures/{figname}")
plt.show()

