import matplotlib.pyplot as plt
import numpy as np
from flow import *
from utils import *
import get_data as gd
import gc

import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())

processor = gd.DataProcessor()
path = "/root/work/truth/signal/*npz"
processor.load_files(path)

(
    CGLMP,
    Higgs,
    LeadLep,
    LepM,
    LepP,
    NuM,
    NuP,
    MET,
    Wm,
    Wp,
    diLep,
    SubLep,
    Xi,
) = processor.files

lep_kin = processor.process_lep(LepP, LepM)
print("lep_kin shape:", lep_kin.shape)
# lep_kin.head(5)

# observed (Y)
nu_kin = processor.process_MET(MET)
print("MET_kin shape:", nu_kin.shape)
# nu_kin.head(5)

# interest (X)
dinu_kin = processor.process_dinu(NuP, NuM)
print("dinu_kin shape:", dinu_kin.shape)
# dinu_kin.head(5)

CGLMP_kin = processor.process_CGLMP(CGLMP)
print("CGLMP shape:", CGLMP_kin.shape)
# CGLMP_kin.head(5)

del processor  # Clear the instance
gc.collect()

from sklearn.model_selection import train_test_split

indices_arr = np.arange(1e4, dtype=int)

lepton_features = [
    "lep_p_E",
    "lep_p_px",
    "lep_p_py",
    "lep_p_pz",
    "lep_m_E",
    "lep_m_px",
    "lep_m_py",
    "lep_m_pz",
]
neutrino_features = [
    "di_nu_E",
    "di_nu_px",
    "di_nu_py",
    "di_nu_pz",
]


def reshape_features_lep(inputs, features):
    outputs = np.stack([inputs[features[0:4]], inputs[features[4:8]]], axis=1)
    return outputs


def reshape_features_dinu(inputs, features):
    outputs = np.expand_dims(inputs[features[0::]].to_numpy(), axis=1)
    return outputs


train_indices, temp_indices = train_test_split(
    indices_arr.flatten(), train_size=0.8, test_size=0.2, random_state=42
)
valid_indices, test_indices = train_test_split(
    temp_indices, train_size=0.5, test_size=0.5, random_state=42
)

lep_train = reshape_features_lep(lep_kin.iloc[train_indices], lepton_features)
lep_valid = reshape_features_lep(lep_kin.iloc[valid_indices], lepton_features)
lep_test = reshape_features_lep(lep_kin.iloc[test_indices], lepton_features)

# interest (X)
dinu_train = np.expand_dims(dinu_kin.to_numpy()[train_indices], axis=1)
dinu_valid = np.expand_dims(dinu_kin.to_numpy()[valid_indices], axis=1)
dinu_test = np.expand_dims(dinu_kin.to_numpy()[test_indices], axis=1)

# observed (Y)
nu_train = np.expand_dims(nu_kin.to_numpy()[train_indices][:, 1:3], axis=1)
nu_valid = np.expand_dims(nu_kin.to_numpy()[valid_indices][:, 1:3], axis=1)
nu_test = np.expand_dims(nu_kin.to_numpy()[test_indices][:, 1:3], axis=1)


def stack_parts(input1, input2):
    outputs = np.concatenate([input1.flatten(), input2.flatten()], axis=0)
    return outputs


# print(f"Training data shape: {lep_train.shape}\nTesting data shape: {lep_test.shape}\nValidation data shape: {lep_valid.shape}")
train_x = stack_parts(lep_train, dinu_train)
valid_x = stack_parts(lep_valid, dinu_valid)
test_x = stack_parts(lep_test, dinu_test)
train_y = stack_parts(lep_train, nu_train)
valid_y = stack_parts(lep_valid, nu_valid)
test_y = stack_parts(lep_test, nu_test)

print(
    f"X (Interest)\nTraining data shape: {train_x.shape}\nTesting data shape: {test_x.shape}\nValidation data shape: {valid_x.shape}"
)
print(
    f"Y (Observed)\nTraining data shape: {train_y.shape}\nTesting data shape: {test_y.shape}\nValidation data shape: {valid_y.shape}"
)
print()

del (
    train_indices,
    temp_indices,
    valid_indices,
    test_indices,
    lep_train,
    lep_valid,
    lep_test,
    nu_train,
    nu_valid,
    nu_test,
)
gc.collect()

# interest (momentum)
x_dim = 2 * 4 + 4
# observed (MET)
y_dim = 2 * 4 + 2
z_dim = 2 * 4 + 4

tot_dim = y_dim + z_dim
pad_dim = tot_dim - x_dim

# Preprocess
y_hat = np.zeros((train_y.reshape(-1, y_dim).shape[0], y_dim))
for i in range(train_y.reshape(-1, y_dim).shape[0]):
    arr = train_y.reshape(-1, y_dim)[i]
    y_hat[i] = np.pad(
        arr, (0, y_dim - arr.shape[0]), mode="constant", constant_values=0
    )

## Pad data
X = train_x.reshape((-1, x_dim))
pad_x = np.zeros((X.shape[0], pad_dim))
x = np.concatenate([X, pad_x], axis=-1).astype("float32")
z = np.random.multivariate_normal([0.0] * z_dim, np.eye(z_dim), X.shape[0]).astype(
    "float32"
)
y = np.concatenate([z, y_hat], axis=-1).astype("float32")

n_sample = X.shape[0]
n_data = n_sample * train_y.flatten().shape[0]
n_couple_layer = 3
n_hid_layer = 3
n_hid_dim = 512

n_batch = 512
n_epoch = 50
n_display = 10

# Make dataset generator
x_data = tf.data.Dataset.from_tensor_slices(x)
y_data = tf.data.Dataset.from_tensor_slices(y)
dataset = (
    tf.data.Dataset.zip((x_data, y_data))
    .shuffle(buffer_size=X.shape[0])
    .batch(n_batch, drop_remainder=True)
    .repeat()
)

model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name="NVP")
x = tfk.Input((tot_dim,))
model(x)
model.summary()


class Trainer(tfk.Model):
    def __init__(
        self,
        model,
        x_dim,
        y_dim,
        z_dim,
        tot_dim,
        n_couple_layer,
        n_hid_layer,
        n_hid_dim,
        shuffle_type="reverse",
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.tot_dim = tot_dim
        self.x_pad_dim = tot_dim - x_dim
        self.y_pad_dim = tot_dim - (y_dim + z_dim)
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type

        self.w1 = 5.0
        self.w2 = 1.0
        self.w3 = 10.0
        self.loss_factor = 1.0
        self.loss_fit = MSE
        self.loss_latent = MMD_multiscale
        self.loss_backward = MMD_multiscale

    def train_step(self, data):
        x_data, y_data = data
        x = x_data[:, : self.x_dim]
        y = y_data[:, -self.y_dim :]
        z = y_data[:, : self.z_dim]
        y_short = tf.concat([z, y], axis=-1)

        # Forward loss
        with tf.GradientTape() as tape:
            y_out = self.model(x_data)
            pred_loss = self.w1 * self.loss_fit(
                y_data[:, self.z_dim :], y_out[:, self.z_dim :]
            )  # [zeros, y] <=> [zeros, yhat]
            output_block_grad = tf.concat(
                [y_out[:, : self.z_dim], y_out[:, -self.y_dim :]], axis=-1
            )  # take out [z, y] only (not zeros)
            latent_loss = self.w2 * self.loss_latent(
                y_short, output_block_grad
            )  # [z, y] <=> [zhat, yhat]
            forward_loss = pred_loss + latent_loss
        grads_forward = tape.gradient(forward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_forward, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            x_rev = self.model.inverse(y_data)
            rev_loss = self.w3 * self.loss_factor * self.loss_backward(x_rev, x_data)
        grads_backward = tape.gradient(rev_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads_backward, self.model.trainable_weights)
        )

        total_loss = forward_loss + latent_loss + rev_loss
        return {
            "total_loss": total_loss,
            "forward_loss": forward_loss,
            "latent_loss": latent_loss,
            "rev_loss": rev_loss,
        }

    def test_step(self, data):
        x_data, y_data = data
        return NotImplementedError


trainer = Trainer(
    model, x_dim, y_dim, z_dim, tot_dim, n_couple_layer, n_hid_layer, n_hid_dim
)
trainer.compile(optimizer="Adam")

LossFactor = UpdateLossFactor(n_epoch)
logger = NBatchLogger(n_display, n_epoch)
hist = trainer.fit(
    dataset,
    batch_size=n_batch,
    epochs=n_epoch,
    steps_per_epoch=n_data // n_batch,
    callbacks=[logger, LossFactor],
    verbose=2,
)

fig, ax = plt.subplots(1, facecolor="white", figsize=(8, 5))
ax.plot(hist.history["total_loss"], "k.-", label="total_loss")
ax.plot(hist.history["forward_loss"], "b.-", label="forward_loss")
ax.plot(hist.history["latent_loss"], "g.-", label="latent_loss")
ax.plot(hist.history["rev_loss"], "r.-", label="inverse_loss")
plt.legend()
plt.savefig("loss.png")
# plt.show()
plt.close()

y_hat = np.zeros((test_y.reshape(-1, y_dim).shape[0], y_dim))
for i in range(test_y.reshape(-1, y_dim).shape[0]):
    arr = test_y.reshape(-1, y_dim)[i]
    y_hat[i] = np.pad(
        arr, (0, y_dim - arr.shape[0]), mode="constant", constant_values=0
    )

z = np.random.multivariate_normal([1.0] * z_dim, np.eye(z_dim), y_hat.shape[0])
y = np.concatenate([z, y_hat], axis=-1).astype("float32")
x_pred = model.inverse(y).numpy()
pz_pred = x_pred[:, -1]
E_pred = x_pred[:, -4]
x_truth = test_x.reshape(-1, x_dim)
pz_truth = x_truth[:, -1]
E_truth = x_truth[:, -4]

plt.plot(pz_truth, pz_pred, ".", alpha=0.3, color="purple")
plt.title(r"$p^{miss}_{z}$")
plt.xlabel("truth")
plt.ylabel("pred")
plt.savefig("pz.png")
# plt.show()
plt.close()

plt.plot(E_truth, E_pred, ".", alpha=0.3, color="purple")
plt.title(r"$E^{miss}$")
plt.xlabel("truth")
plt.ylabel("pred")
plt.savefig("energy.png")
# plt.show()
plt.close()


print("Finish!")
