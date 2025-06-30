import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from tqdm.auto import tqdm

from b_pure_metoropolis_2d_ising import ising_energy
from onsager import onsager_energy
from utils import load_np, save_np

# --- 1. パラメータ設定 ---
L = 20  # 格子のサイズ
J = 1.0  # 交換結合定数 (強磁性)
h = 0.0  # 外部磁場
kT = 2.0  # 温度
N_SAMPLES = 2000  # 学習データのサンプル数
N_STEPS_MC = L * L * 100  # メトロポリス法のステップ数

# D3PMのパラメータ
T = 1000  # 拡散のタイムステップ数
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
N_EPOCHS = 30

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")


# --- 2. Isingデータ生成 (メトロポリス法) ---
def generate_ising_data(n_samples, L, J, h, kT):
    """
    トロポリス法でIsing配位のデータセットを生成する
    """
    input_meta: dict[str, str | int | float] = dict(
        name="ising_data", n_samples=n_samples, L=L, J=J, h=h, kT=kT
    )
    loaded = load_np(input_meta)
    if loaded is not None:
        print("Loaded existing Ising dataset.")
        return tf.convert_to_tensor(loaded, dtype=tf.int32)

    dataset = np.empty((n_samples, L, L), dtype=np.int8)

    for i in tqdm(range(n_samples), desc="Generating Ising data"):
        # spins = np.random.choice([-1, 1], size=(L, L))
        spins = np.ones((L, L), dtype=np.int8)
        for _ in range(N_STEPS_MC):
            x, y = np.random.randint(0, L, size=2)
            s = spins[x, y]
            neighbors = (
                spins[(x + 1) % L, y]
                + spins[(x - 1) % L, y]
                + spins[x, (y + 1) % L]
                + spins[x, (y - 1) % L]
            )
            delta_E = 2 * s * (J * neighbors + h)
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / kT):
                spins[x, y] = -s
        dataset[i] = spins

    dataset[dataset == -1] = 0
    save_np(dataset, input_meta)

    return tf.convert_to_tensor(dataset, dtype=tf.int32)


# --- 3. 簡易CNNモデルの定義 ---
class SimpleCNN_TF(keras.Model):
    def __init__(self, n_steps=T, **kwargs):
        super().__init__(**kwargs)
        self.time_emb = layers.Embedding(n_steps, 32)

        # 時刻埋め込みを畳み込み層のチャネル数に合わせるための全結合層
        self.time_proj = layers.Dense(64)  # conv1の出力チャネル(64)に合わせる

        self.conv1 = layers.Conv2D(64, kernel_size=3, padding="same", activation="tanh")
        self.conv2 = layers.Conv2D(64, kernel_size=3, padding="same", activation="tanh")
        self.conv3 = layers.Conv2D(2, kernel_size=3, padding="same")

    def call(self, x, t):
        x = tf.expand_dims(x, -1)
        x = tf.cast(x, tf.float32)

        # 時刻埋め込みを計算し、チャネル数に合わせる
        t_emb = self.time_emb(t)  # (B, 32)
        t_emb = self.time_proj(t_emb)  # (B, 64)

        # 画像の形状 (B, L, L, C) に合わせて次元を追加
        t_emb = tf.expand_dims(tf.expand_dims(t_emb, 1), 1)  # (B, 1, 1, 64)

        # 最初の畳み込み
        x = self.conv1(x)  # x の shape は (B, L, L, 64)
        x = x + t_emb

        # 2番目の畳み込み以降
        x = self.conv2(x)
        out = self.conv3(x)
        return out


# --- 4. D3PMロジックの実装---
class D3PM_TF(keras.Model):
    def __init__(self, denoise_model, n_steps=T, **kwargs):
        super().__init__(**kwargs)
        self.denoise_model = denoise_model
        self.T = n_steps
        self.K = 2

        self.betas = tf.linspace(1e-4, 0.02, self.T)

        # 遷移確率行列の計算
        q_xt_x0_probas_list = []
        log_alpha_bar = 0.0
        for t in range(self.T):
            beta_t = self.betas[t]
            log_alpha_t = tf.math.log(1 - beta_t)
            log_alpha_bar += log_alpha_t
            diag = tf.exp(log_alpha_bar)
            off_diag = (1 - diag) / self.K

            # 対角行列と非対角行列を作成し、足し合わせる
            matrix = off_diag * tf.ones((self.K, self.K), dtype=tf.float32) + (
                diag - off_diag
            ) * tf.eye(self.K, dtype=tf.float32)
            q_xt_x0_probas_list.append(matrix)

        # 最後にリストをスタックして1つのテンソルにする
        self.q_xt_x0_probas = tf.stack(q_xt_x0_probas_list)

    def q_sample(self, x0, t):
        """
        x0 と t から xt をサンプリングする (順方向過程)
        """
        # probas: (B, K, K) だが einsum で使うには次元調整が必要
        probas = tf.gather(self.q_xt_x0_probas, t)  # (B, K, K)
        probas = tf.reshape(probas, [-1, 1, 1, self.K, self.K])  # (B, 1, 1, K, K)
        x0_onehot = tf.one_hot(x0, depth=self.K, dtype=tf.float32)  # (B, L, L, K)
        x0_onehot = tf.expand_dims(x0_onehot, -2)  # (B, L, L, 1, K)
        xt_probas = tf.matmul(x0_onehot, probas)  # (B, L, L, 1, K)
        xt_probas = tf.squeeze(xt_probas, -2)  # (B, L, L, K)

        shape = tf.shape(xt_probas)
        B, L_dim, _, K_dim = (shape[0], shape[1], shape[2], shape[3])  # type: ignore

        xt_probas_flat = tf.reshape(xt_probas, [-1, K_dim])
        # 0の確率を避けるため微小値を加える
        xt_logits_flat = tf.math.log(tf.clip_by_value(xt_probas_flat, 1e-8, 1.0))
        xt_flat = tf.random.categorical(xt_logits_flat, 1)
        # 元のShapeに戻す
        xt = tf.reshape(xt_flat, [B, L_dim, L_dim])

        return xt

    def q_posterior_probas(self, x0_hat_logits, xt, t):
        beta_t = tf.gather(self.betas, t)
        beta_t = tf.reshape(beta_t, [-1, 1, 1, 1])

        x0_hat_probas = tf.nn.softmax(x0_hat_logits, axis=-1)
        x0_hat_probas = tf.transpose(x0_hat_probas, perm=[0, 3, 1, 2])

        xt_onehot = tf.transpose(
            tf.one_hot(xt, depth=self.K, dtype=tf.float32), perm=[0, 3, 1, 2]
        )

        term1 = (1 - beta_t) * x0_hat_probas
        term2 = (beta_t / self.K) * tf.reduce_sum(x0_hat_probas, axis=1, keepdims=True)

        numerator = (term1 + term2) * xt_onehot

        norm = tf.reduce_sum(numerator, axis=1, keepdims=True)
        xtm1_probas = numerator / tf.clip_by_value(norm, 1e-8, tf.float32.max)

        return tf.transpose(xtm1_probas, perm=[0, 2, 3, 1])

    def p_sample(self, xt, t_int):
        shape = tf.shape(xt)
        B, L_dim = shape[0], shape[1]  # type: ignore
        t = tf.fill([B], t_int)
        x0_hat_logits = self.denoise_model(xt, t)
        xtm1_probas = self.q_posterior_probas(x0_hat_logits, xt, t)
        xtm1_logits = tf.math.log(tf.clip_by_value(xtm1_probas, 1e-8, 1.0))
        xtm1 = tf.random.categorical(tf.reshape(xtm1_logits, [-1, self.K]), 1)
        return tf.reshape(xtm1, [B, L_dim, L_dim])

    def sample(self, n_samples):
        xt = tf.random.uniform(
            shape=(n_samples, L, L), minval=0, maxval=self.K, dtype=tf.int32
        )
        for t in tqdm(reversed(range(self.T)), desc="Sampling", total=self.T):
            xt = self.p_sample(xt, t)

        generated = tf.where(xt == 0, -1, 1)
        return generated.numpy()

    def call(self, inputs, training=False):
        x0 = inputs
        B = tf.shape(x0)[0]  # type: ignore
        t = tf.random.uniform(shape=(B,), minval=0, maxval=self.T, dtype=tf.int32)
        xt = self.q_sample(x0, t)
        x0_hat_logits = self.denoise_model(xt, t)
        return x0_hat_logits


# --- 5. 学習とサンプリングの実行 ---
def main():
    ising_data = generate_ising_data(N_SAMPLES, L, J, h, kT)
    dataset = tf.data.Dataset.from_tensor_slices(ising_data)
    dataloader = (
        dataset.batch(BATCH_SIZE)
        .shuffle(buffer_size=N_SAMPLES)
        .prefetch(tf.data.AUTOTUNE)
    )

    cnn_model = SimpleCNN_TF(n_steps=T)
    # cnn_model.build(input_shape=(None, L, L))  # 入力の形状を指定
    # cnn_model.summary()
    d3pm = D3PM_TF(denoise_model=cnn_model, n_steps=T)

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(x0):
        with tf.GradientTape() as tape:
            x0_hat_logits = d3pm(x0, training=True)
            loss = loss_fn(x0, x0_hat_logits)

        gradients = tape.gradient(loss, d3pm.trainable_variables)
        assert gradients is not None, "Gradients are None, check the model and inputs."
        optimizer.apply_gradients(zip(gradients, d3pm.trainable_variables))
        return loss

    losses = []
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False):
            loss = train_step(batch)
            epoch_loss += loss  # type: ignore # TODO: handle type
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")

    n_generate = 100
    generated_samples = d3pm.sample(n_generate)

    # --- 6. 結果の表示 ---
    onsager_energy_value = onsager_energy(kT, J)
    print(f"Onsager energy at kT={kT}, L={L}: {onsager_energy_value:.4f}")
    data_energy = np.mean(ising_energy(ising_data.numpy(), J, h) / L**2)  # type: ignore
    print(f"Data energy: {data_energy:.4f}")
    rec_energy = np.mean(ising_energy(generated_samples, J, h) / L**2)
    print(f"Reconstructed energy: {rec_energy:.4f}")

    fig, axes = plt.subplots(2, 5, figsize=(5 * 2.5, 5))
    fig.suptitle(f"D3PM Sampling (kT={kT}, L={L})", fontsize=16)

    original_numpy = ising_data.numpy()  # type: ignore
    random_indices = np.random.choice(n_generate, 5, replace=False)
    for i, idx in enumerate(random_indices):
        ax = axes[0, i]
        sample_data = original_numpy[idx]
        sample_data[sample_data == 0] = -1
        ax.imshow(sample_data, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Original Data")

    for i, idx in enumerate(random_indices):
        ax = axes[1, i]
        ax.imshow(generated_samples[idx], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Generated Data")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse Categorical Crossentropy")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
