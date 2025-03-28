# %%
import time
t_start = time.time()

import sys
import os
import glob

import pandas as pd
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt

HOME_PATH = os.path.abspath("/root/work/QE-mls")  # path to QE_mls
sys.path.append(HOME_PATH + "/qe")
print("Home path:", HOME_PATH)
# custom modules
import paper_plot as pp
import full_data as data

# %%
# ML libraries need to be after other packages imported
import tensorflow as tf
print("Tensorflow:", tf.__version__)
import tensorflow_probability as tfp
print("Tensorflow Probability:", tfp.__version__)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# GPU(s) setups
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu_ind = 0
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[gpu_ind], 'GPU')
        # Enable memory growth for the first GPU
        tf.config.experimental.set_memory_growth(gpus[gpu_ind], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

import keras_tuner as kt
from keras.saving import register_keras_serializable
from sklearn.model_selection import train_test_split

# %%
# ROOT need to be imported after tensorflow
import ROOT
from ROOT import TLorentzVector, TVector3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow information messages
os.environ["TF_CUDNN_LOGDEST"] = "0"  # suppress cuDNN log output
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# some global variables
plot = pp.Plotter()
WORKERS = 16
SEED = 42  # set random seed (global variable)
GEV = 1e-3
# training variables
SIGMA_LST = [1.0, 5.0, 10.0, 100.0, 400.0, 800.0]
BATCH_SIZE = 256
EPOCHS = 2048
# LEARNING_RATE = 1e-5
# Make a directory
dir_name = HOME_PATH + "/12th_trial/reg_main/ww_resregressor_result12_hytune/"
savedmodel_path = dir_name + "saved_model"
name = "ww_resregressor"

# %%
# reco
lead_lep = data.Lead_lep()
sublead_lep = data.Sublead_lep()
met = data.Met()

# truth
truth_lead_lep = data.Truth_lead_lep()
truth_sublead_lep = data.Truth_sublead_lep()
Truth_dilep = data.Truth_dilep()
truth_met = data.Truth_met()

# target Ws
w_lead = data.Lead_w()
w_sublead = data.Sublead_w()
mc_weight = data.MC_weight()

# %%
# define preselection cuts
cut_pre_pt_lead = lead_lep.pt > 22  # 22 ~ 40 (SY)
print(len(lead_lep.energy) - np.sum(cut_pre_pt_lead * 1))
cut_pre_pt_sub = sublead_lep.pt > 15
print(len(lead_lep.energy) - np.sum(cut_pre_pt_sub * 1))
cut_pre_dilep_m = (
    np.sqrt(
        np.square(lead_lep.energy + sublead_lep.energy)
        - np.square(lead_lep.px + sublead_lep.px)
        - np.square(lead_lep.py + sublead_lep.py)
        - np.square(lead_lep.pz + sublead_lep.pz)
    )
    > 10
)
print(len(lead_lep.energy) - np.sum(cut_pre_dilep_m * 1))
cut_pre_pt_miss = met.pt > 20
print(len(lead_lep.energy) - np.sum(cut_pre_pt_miss * 1))
PRE_CUT = cut_pre_pt_lead & cut_pre_pt_sub & cut_pre_dilep_m & cut_pre_pt_miss
print(len(lead_lep.energy) - np.sum(PRE_CUT * 1))
del (cut_pre_pt_lead, cut_pre_pt_sub, cut_pre_dilep_m, cut_pre_pt_miss)

# %%
def eta(px, py, pz):
    pt = np.sqrt(np.square(px) + np.square(py))
    return np.arcsinh(np.divide(pz, pt))

def phi(px, py):
    return np.arctan2(py, px)

def deta(eta1, eta2):
    return np.abs(eta1 - eta2)

def dphi_pi(phi1, phi2):
    phi_diff = phi1 - phi2
    phi_diff = np.where(phi_diff < 0.0, -phi_diff, phi_diff)
    phi_diff = np.where(phi_diff > 2.0 * np.pi, phi_diff - 2.0 * np.pi, phi_diff)
    phi_diff = np.where(phi_diff >= np.pi, 2.0 * np.pi - phi_diff, phi_diff)
    return np.divide(phi_diff, np.pi)

# %%
# truth
truth_lead_lep_px = pd.DataFrame(truth_lead_lep.px[PRE_CUT])
truth_lead_lep_py = pd.DataFrame(truth_lead_lep.py[PRE_CUT])
truth_lead_lep_pz = pd.DataFrame(truth_lead_lep.pz[PRE_CUT])
truth_lead_lep_energy = pd.DataFrame(truth_lead_lep.energy[PRE_CUT])
truth_lead_lep_p4 = pd.concat(
    [truth_lead_lep_px, truth_lead_lep_py, truth_lead_lep_pz, truth_lead_lep_energy],
    axis=1,
).to_numpy()
truth_sublead_lep_px = pd.DataFrame(truth_sublead_lep.px[PRE_CUT])
truth_sublead_lep_py = pd.DataFrame(truth_sublead_lep.py[PRE_CUT])
truth_sublead_lep_pz = pd.DataFrame(truth_sublead_lep.pz[PRE_CUT])
truth_sublead_lep_energy = pd.DataFrame(truth_sublead_lep.energy[PRE_CUT])
truth_sublead_lep_p4 = pd.concat(
    [truth_sublead_lep_px, truth_sublead_lep_py, truth_sublead_lep_pz, truth_sublead_lep_energy],
    axis=1,
).to_numpy()
truth_dilep_dphi = pd.DataFrame(truth_lead_lep.phi[PRE_CUT] - truth_sublead_lep.phi[PRE_CUT])
truth_dilep_deta = pd.DataFrame(truth_lead_lep.eta[PRE_CUT] - truth_sublead_lep.eta[PRE_CUT])
# reco
lead_lep_px = pd.DataFrame(lead_lep.px[PRE_CUT])
lead_lep_py = pd.DataFrame(lead_lep.py[PRE_CUT])
lead_lep_pz = pd.DataFrame(lead_lep.pz[PRE_CUT])
lead_lep_energy = pd.DataFrame(lead_lep.energy[PRE_CUT])
lead_lep_p4 = pd.concat(
    [lead_lep_px, lead_lep_py, lead_lep_pz, lead_lep_energy],
    axis=1,
).to_numpy()
sublead_lep_px = pd.DataFrame(sublead_lep.px[PRE_CUT])
sublead_lep_py = pd.DataFrame(sublead_lep.py[PRE_CUT])
sublead_lep_pz = pd.DataFrame(sublead_lep.pz[PRE_CUT])
sublead_lep_energy = pd.DataFrame(sublead_lep.energy[PRE_CUT])
sublead_lep_p4 = pd.concat(
    [sublead_lep_px, sublead_lep_py, sublead_lep_pz, sublead_lep_energy],
    axis=1,
).to_numpy()
met_px = pd.DataFrame(met.px[PRE_CUT])
met_py = pd.DataFrame(met.py[PRE_CUT])
met_phi = pd.DataFrame(met.phi[PRE_CUT])

# advanced features
ll_phi = phi(lead_lep.px[PRE_CUT] + sublead_lep.px[PRE_CUT], lead_lep.py[PRE_CUT] + sublead_lep.py[PRE_CUT])
ll_eta = eta(lead_lep.px[PRE_CUT] + sublead_lep.px[PRE_CUT], lead_lep.py[PRE_CUT] + sublead_lep.py[PRE_CUT], lead_lep.pz[PRE_CUT] + sublead_lep.pz[PRE_CUT])
ll_deta = deta(lead_lep.eta[PRE_CUT], sublead_lep.eta[PRE_CUT])
ll_dphi = dphi_pi(lead_lep.phi[PRE_CUT], sublead_lep.phi[PRE_CUT])
metl0_dphi = dphi_pi(met.phi[PRE_CUT], lead_lep.phi[PRE_CUT])
metl1_dphi = dphi_pi(met.phi[PRE_CUT], sublead_lep.phi[PRE_CUT])
metll_dphi = dphi_pi(met.phi[PRE_CUT], ll_phi)

# Concatenate along axis=1 to get N×7 shape
two_part_features = np.vstack([ll_phi, ll_eta, ll_deta, ll_dphi, metl0_dphi, metl1_dphi, metll_dphi]).T
print(f"Shape of two_part_features: {two_part_features.shape}")

# %%
# Observing variables
obs_kin = np.column_stack(
    (
        lead_lep_px,
        lead_lep_py,
        lead_lep_pz,
        lead_lep_energy,
        sublead_lep_px,
        sublead_lep_py,
        sublead_lep_pz,
        sublead_lep_energy,
        met_px,
        met_py,
    )
)
print("obs_kin shape:", obs_kin.shape)

# %%
# Interesting variables
w_lead_px = pd.DataFrame(w_lead.px[PRE_CUT])
w_lead_py = pd.DataFrame(w_lead.py[PRE_CUT])
w_lead_pz = pd.DataFrame(w_lead.pz[PRE_CUT])
w_lead_e = pd.DataFrame(w_lead.energy[PRE_CUT])
w_sublead_px = pd.DataFrame(w_sublead.px[PRE_CUT])
w_sublead_py = pd.DataFrame(w_sublead.py[PRE_CUT])
w_sublead_pz = pd.DataFrame(w_sublead.pz[PRE_CUT])
w_sublead_e = pd.DataFrame(w_sublead.energy[PRE_CUT])
w_lead_m = pd.DataFrame(w_lead.m[PRE_CUT])
w_sublead_m = pd.DataFrame(w_sublead.m[PRE_CUT])

int_kin = np.concatenate(
    [w_lead_px, w_lead_py, w_lead_pz, w_lead_e, w_sublead_px, w_sublead_py, w_sublead_pz, w_sublead_e, w_lead_m, w_sublead_m],
    axis=-1,
)
print("int_kin shape:", int_kin.shape)
print(type(int_kin))

# %%
# Train test split
indices_arr = np.arange(int_kin.shape[0], dtype="int")
train_indices, temp_indices = train_test_split(
    indices_arr.flatten(), train_size=0.8, test_size=0.2, random_state=SEED, shuffle=True
)
valid_indices, test_indices = train_test_split(
    temp_indices, train_size=0.5, test_size=0.5, random_state=SEED
)

train_x = obs_kin[train_indices]
test_x = obs_kin[test_indices]
valid_x = obs_kin[valid_indices]
train_y = int_kin[train_indices]
test_y = int_kin[test_indices]
valid_y = int_kin[valid_indices]

print("Total evts:", len(indices_arr))
print(f"X (Observed)\nTraining data shape: {train_x.shape};\nValidating data shape: {valid_x.shape};\nTesting data shape: {test_x.shape}.")
print(f"Y (Interested)\nTraining data shape: {train_y.shape};\nValidating data shape: {valid_y.shape};\nTesting data shape: {test_y.shape}.")

# %%
# create datasets
train_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.cast(train_x, tf.float32), tf.cast(train_y, tf.float32)))
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.cast(valid_x, tf.float32), tf.cast(valid_y, tf.float32)))
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# %%
# Loss functions
def compute_mmd(x, y, sigma_list=SIGMA_LST):
    x = tf.reshape(x, [-1, 1])
    y = tf.reshape(y, [-1, 1])
    xx = tf.linalg.matmul(x, tf.transpose(x))
    yy = tf.linalg.matmul(y, tf.transpose(y))
    xy = tf.linalg.matmul(x, tf.transpose(y))
    diag_xx = tf.linalg.diag_part(xx)
    diag_yy = tf.linalg.diag_part(yy)
    dxx = tf.expand_dims(diag_xx, 0) + tf.expand_dims(diag_xx, 1) - 2.0 * xx
    dyy = tf.expand_dims(diag_yy, 0) + tf.expand_dims(diag_yy, 1) - 2.0 * yy
    dxy = tf.expand_dims(diag_xx, 1) + tf.expand_dims(diag_yy, 0) - 2.0 * xy
    XX = tf.zeros_like(xx, dtype=tf.float32)
    YY = tf.zeros_like(yy, dtype=tf.float32)
    XY = tf.zeros_like(xy, dtype=tf.float32)
    for sigma in sigma_list:
        XX += tf.exp(-dxx / (2.0 * tf.square(sigma)))
        YY += tf.exp(-dyy / (2.0 * tf.square(sigma)))
        XY += tf.exp(-dxy / (2.0 * tf.square(sigma)))
    mmd_value = tf.reduce_mean(XX + YY - 2.0 * XY)
    return mmd_value

def w_mass_loss_fn(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    w0_4Vect = y_pred[..., :4]
    w0_true_mass = y_true[..., 8]
    w1_4Vect = y_pred[..., 4:8]
    w1_true_mass = y_true[..., 9]
    w0_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(w0_4Vect[..., 3]) - tf.reduce_sum(tf.square(w0_4Vect[..., :3]), axis=-1)), 1e-10))
    w1_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(w1_4Vect[..., 3]) - tf.reduce_sum(tf.square(w1_4Vect[..., :3]), axis=-1)), 1e-10))
    w0_mae = tf.reduce_mean(tf.abs(w0_mass - w0_true_mass))
    w1_mae = tf.reduce_mean(tf.abs(w1_mass - w1_true_mass))
    return w0_mae, w1_mae

def w_mass_mmd_loss_fn(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    w0_4Vect = y_pred[..., :4]
    w0_true_mass = y_true[..., 8]
    w1_4Vect = y_pred[..., 4:8]
    w1_true_mass = y_true[..., 9]
    w0_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(w0_4Vect[..., 3]) - tf.reduce_sum(tf.square(w0_4Vect[..., :3]), axis=-1)), 1e-10))
    w1_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(w1_4Vect[..., 3]) - tf.reduce_sum(tf.square(w1_4Vect[..., :3]), axis=-1)), 1e-10))
    mmd0 = compute_mmd(w0_mass, w0_true_mass)
    mmd1 = compute_mmd(w1_mass, w1_true_mass)
    return mmd0, mmd1

def nu_mass_loss_fn(x_batch, y_pred):
    _epsilon = 1e-10
    x_batch = tf.cast(x_batch, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    n0_4vect = y_pred[..., :4] - x_batch[..., :4]
    n1_4vect = y_pred[..., 4:8] - x_batch[..., 4:8]
    nu0_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(n0_4vect[..., 3]) - tf.reduce_sum(tf.square(n0_4vect[..., :3]), axis=-1)), _epsilon))
    nu1_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(n1_4vect[..., 3]) - tf.reduce_sum(tf.square(n1_4vect[..., :3]), axis=-1)), _epsilon))
    return tf.reduce_mean(nu0_mass + nu1_mass)

def dinu_pt_loss_fn(x_batch, y_pred):
    _epsilon = 1e-10
    x_batch = tf.cast(x_batch, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    n0_4vect = y_pred[..., :4] - x_batch[..., :4]
    n1_4vect = y_pred[..., 4:8] - x_batch[..., 4:8]
    nn_4vect = n0_4vect + n1_4vect
    nn_px = nn_4vect[..., 0]
    nn_py = nn_4vect[..., 1]
    nn_px_diff = tf.math.maximum(tf.abs(nn_px - x_batch[..., 8]), _epsilon)
    nn_py_diff = tf.math.maximum(tf.abs(nn_py - x_batch[..., 9]), _epsilon)
    return tf.reduce_mean(nn_px_diff + nn_py_diff)

def higgs_mass_loss_fn(y_pred):
    _epsilon = 1e-10
    y_pred = tf.cast(y_pred, tf.float32)
    w0_4Vect = y_pred[..., :4]
    w1_4Vect = y_pred[..., 4:8]
    higgs_4Vect = w0_4Vect + w1_4Vect
    higgs_mass = tf.sqrt(tf.math.maximum(tf.abs(tf.square(higgs_4Vect[..., 3]) - tf.reduce_sum(tf.square(higgs_4Vect[..., :3]), axis=-1)), _epsilon))
    return tf.reduce_mean(tf.abs(higgs_mass - 125.0)) + _epsilon

def mae_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)[:, 0:8]
    y_pred = tf.cast(y_pred, tf.float32)[:, 0:8]
    return tf.reduce_mean(tf.keras.losses.mae(y_true, y_pred))

def neg_r2_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)[:, 0:8]
    y_pred = tf.cast(y_pred, tf.float32)[:, 0:8]
    return (tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) - 1)

# %%
@register_keras_serializable(package="ww_regressor")
class CustomModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.current_epoch = tf.Variable(0.0, trainable=False)
        self.metric_dict = {
            "mae_loss": tf.keras.metrics.Mean(name="mae1_loss"),
            "nu_mass_loss": tf.keras.metrics.Mean(name="nu_mass_loss"),
            "w0_mass_mae_loss": tf.keras.metrics.Mean(name="w0_mass_mae_loss"),
            "w1_mass_mae_loss": tf.keras.metrics.Mean(name="w1_mass_mae_loss"),
            "w_mass_mmd0_loss": tf.keras.metrics.Mean(name="w_mass_mmd0_loss"),
            "w_mass_mmd1_loss": tf.keras.metrics.Mean(name="w_mass_mmd1_loss"),
            "higgs_mass_loss": tf.keras.metrics.Mean(name="higgs_mass_loss"),
            "dinu_pt_loss": tf.keras.metrics.Mean(name="dinu_pt_loss"),
            "neg_r2_loss": tf.keras.metrics.Mean(name="neg_r2_loss"),
            "loss": tf.keras.metrics.Mean(name="loss"),
        }

    def build(self, input_shape):
        """Build the model by delegating to the base_model."""
        self.base_model.build(input_shape)
        super().build(input_shape)  # Call the parent class's build to finalize

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config["base_model"] = tf.keras.utils.serialize_keras_object(self.base_model)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        base_model_config = config.pop("base_model", None)
        for key in ["name", "trainable", "dtype"]:
            if key in config:
                config.pop(key)
        if base_model_config is not None:
            base_model = tf.keras.utils.deserialize_keras_object(base_model_config, custom_objects=custom_objects)
        else:
            raise ValueError("No 'base_model' found in config—cannot rebuild the original model.")
        return cls(base_model=base_model)

    def compile(self, optimizer, loss_weights=None, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        default_weights = {
            "mae": 1.0, "nu_mass": 0.0, "higgs_mass": 0.0, "w0_mass_mae": 0.0, "w1_mass_mae": 0.0,
            "w_mass_mmd0": 0.0, "w_mass_mmd1": 0.0, "dinu_pt": 0.0, "neg_r2": 0.0, "fused_output": 0.0,
        }
        self.loss_weights = {**default_weights, **(loss_weights or {})}

    @property
    def metrics(self):
        return list(self.metric_dict.values())

    def _compute_losses(self, x, y, predictions):
        outputs = predictions
        losses = {
            "mae": mae_loss_fn(y, outputs),
            "nu_mass": nu_mass_loss_fn(x, outputs),
            "higgs_mass": higgs_mass_loss_fn(outputs),
            "w0_mass_mae": w_mass_loss_fn(y, outputs)[0],
            "w1_mass_mae": w_mass_loss_fn(y, outputs)[1],
            "w_mass_mmd0": w_mass_mmd_loss_fn(y, outputs)[0],
            "w_mass_mmd1": w_mass_mmd_loss_fn(y, outputs)[1],
            "dinu_pt": dinu_pt_loss_fn(x, outputs),
            "neg_r2": neg_r2_loss_fn(y, outputs),
        }
        total_loss = tf.add_n([self.loss_weights[name] * loss for name, loss in losses.items()])
        return total_loss, losses

    def _update_metrics(self, total_loss, losses):
        self.metric_dict["loss"].update_state(total_loss)
        for name, loss in losses.items():
            self.metric_dict[f"{name}_loss"].update_state(loss)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            total_loss, losses = self._compute_losses(x, y, predictions)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._update_metrics(total_loss, losses)
        return {name: metric.result() for name, metric in self.metric_dict.items()}

    def test_step(self, data):
        x, y = data
        predictions = self(x, training=False)
        total_loss, losses = self._compute_losses(x, y, predictions)
        self._update_metrics(total_loss, losses)
        return {name: metric.result() for name, metric in self.metric_dict.items()}

# %%
@register_keras_serializable(package="nu_regressor")
class WBosonFourVectorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WBosonFourVectorLayer, self).__init__(**kwargs)

    def call(self, inputs):
        lep0, lep1, nu_3mom = inputs
        nu0_3mom = nu_3mom[..., :3]
        nu1_3mom = nu_3mom[..., 3:]
        nu0_p_squared = tf.reduce_sum(tf.square(nu0_3mom), axis=-1, keepdims=True)
        nu1_p_squared = tf.reduce_sum(tf.square(nu1_3mom), axis=-1, keepdims=True)
        nu0_energy = tf.sqrt(tf.maximum(nu0_p_squared, 1e-10))
        nu1_energy = tf.sqrt(tf.maximum(nu1_p_squared, 1e-10))
        nu0_4vect = tf.concat([nu0_3mom, nu0_energy], axis=-1)
        nu1_4vect = tf.concat([nu1_3mom, nu1_energy], axis=-1)
        w0_4vect = lep0 + nu0_4vect
        w1_4vect = lep1 + nu1_4vect
        return tf.concat([w0_4vect, w1_4vect], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 8)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# %%
def dense_dropout_block(x, units, activation="swish", dropout_rate=0.0, l2=0.0):
    x = tf.keras.layers.Dense(
        units=units, use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(l2)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x

def residual_block(x, units, activation="swish", dropout_rate=0.0, l2=0.0):
    y = dense_dropout_block(x, units, activation=activation, dropout_rate=dropout_rate, l2=l2)
    y = dense_dropout_block(y, units, activation=activation, dropout_rate=dropout_rate, l2=l2)
    if x.shape[-1] != units:
        x = tf.keras.layers.Dense(units, activation=None, use_bias=False)(x)
    z = tf.keras.layers.Add()([x, y])
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation(activation)(z)
    return z

# %%
def build_hypermodel(hp):
    tf.keras.backend.clear_session()
    # learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    learning_rate = 1e-5
    # dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    dropout_rate = 0.4
    # l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log')
    l2_reg = 1e-4
    units_base = hp.Int('units_base', min_value=32, max_value=512, step=32)

    inputs = tf.keras.layers.Input(shape=(input_shape,), dtype=tf.float32)
    x = inputs
    lep0 = inputs[..., :4]
    lep1 = inputs[..., 4:8]

    for _ in range(2):
        x = residual_block(x, units_base, dropout_rate=dropout_rate, l2=l2_reg)
        x = residual_block(x, units_base * 2, dropout_rate=dropout_rate, l2=l2_reg)
    for _ in range(2):
        x = residual_block(x, units_base * 4, dropout_rate=dropout_rate, l2=l2_reg)
        x = residual_block(x, units_base * 8, dropout_rate=dropout_rate, l2=l2_reg)
    for _ in range(2):
        x = residual_block(x, units_base * 4, dropout_rate=dropout_rate, l2=l2_reg)
        x = residual_block(x, units_base * 2, dropout_rate=dropout_rate, l2=l2_reg)

    x = tf.keras.layers.Dense(16, activation=None, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)

    nu_3mom = tf.keras.layers.Dense(6, activation="linear", kernel_initializer="he_normal")(x)
    neutrino_outputs = WBosonFourVectorLayer()([lep0, lep1, nu_3mom])
    w0_mass = tf.keras.layers.Dense(1, activation="swish", kernel_initializer="he_normal")(x)
    w1_mass = tf.keras.layers.Dense(1, activation="swish", kernel_initializer="he_normal")(x)
    outputs = tf.keras.layers.Concatenate()([neutrino_outputs, w0_mass, w1_mass])
    base_model = tf.keras.Model(inputs=inputs, outputs=[outputs])
    custom_model = CustomModel(base_model)

    custom_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss_weights={
            "mae": 1.0,
            "w_mass_mmd0": 10.0,
            "w_mass_mmd1": 20.0,
            # "mae": hp.Float('mae_weight', min_value=1.0, max_value=10.0, sampling='log'),
            # "w_mass_mmd0": hp.Float('w_mass_mmd0_weight', min_value=1.0, max_value=20.0, sampling='log'),
            # "w_mass_mmd1": hp.Float('w_mass_mmd1_weight', min_value=1.0, max_value=20.0, sampling='log'),
        },
        jit_compile=False, # some error occurs with JIT compilation.... IDK why :(
        steps_per_execution=256,
    )
    return custom_model

# %%
# Define callbacks
class EpochUpdater(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(float(epoch))

class LambdaTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = int(self.model.current_epoch.numpy())
        log_str = f"Epoch {epoch_num}"
        for name, value in logs.items():
            log_str += f"; {name}: {value:.4f}"
        print(log_str)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    EpochUpdater(),
    LambdaTracker(),
]

# Define input shape
input_shape = train_x.shape[-1]

# Initialize the tuner
tuner = kt.Hyperband(
    build_hypermodel,
    objective='val_loss',
    max_epochs=128,
    factor=3,
    directory=dir_name + 'hyperband_dir',
    project_name='ww_regressor12_tuning'
)

# Perform hyperparameter search
tuner.search(
    train_dataset,
    validation_data=valid_dataset,
    epochs=128,
    callbacks=callbacks,
    verbose=0
)

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]

# Build the model with the input shape before loading weights or training
# best_model.build((None, input_shape))  # None for batch size, input_shape for features

# Retrieve and print the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("############################################")
print("Best Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")
print("############################################")

# Train the best model fully
history = best_model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=0,
)

# Print the final loss weights
print("\nModel Loss Weights:")
for name, weight in best_model.loss_weights.items():
    print(f"{name:12s}: {weight:.3f}")

# %%

if os.path.exists(dir_name):
    test_files = glob.glob(dir_name + "*" + name + "*")
    if test_files:
        for file in test_files:
            if os.path.isdir(file):
                import shutil
                shutil.rmtree(file)
            else:
                os.remove(file)
        print(f"Directory {dir_name} cleaned.")
    print(f"Directory {dir_name} exists.")
else:
    os.makedirs(dir_name)
    print(f"Directory {dir_name} created.")

try:
    best_model.save(dir_name + name + ".keras", save_format="keras", overwrite=True)
    tf.saved_model.save(best_model, savedmodel_path)
    print(f"Model saved in {dir_name}.")
except Exception as e:
    print(f"Error: {e}")

# Predict with the best model
pred_y = best_model.predict(test_x)

# %%
# Extract loss values
loss = history.history["loss"]
val_loss = history.history["val_loss"]
mae_loss = history.history["mae_loss"]
val_mae_loss = history.history["val_mae_loss"]
neg_r2_loss = history.history["neg_r2_loss"]
val_neg_r2_loss = history.history["val_neg_r2_loss"]
higgs_mass_loss = history.history["higgs_mass_loss"]
val_higgs_mass_loss = history.history["val_higgs_mass_loss"]
dinu_pt_loss = history.history["dinu_pt_loss"]
val_dinu_pt_loss = history.history["val_dinu_pt_loss"]
nu_mass_loss = history.history["nu_mass_loss"]
val_nu_mass_loss = history.history["val_nu_mass_loss"]
w_mass_mmd0_loss = history.history["w_mass_mmd0_loss"]
val_w_mass_mmd0_loss = history.history["val_w_mass_mmd0_loss"]
w_mass_mmd1_loss = history.history["w_mass_mmd1_loss"]
val_w_mass_mmd1_loss = history.history["val_w_mass_mmd1_loss"]
w0_mass_mae_loss = history.history["w0_mass_mae_loss"]
val_w0_mass_mae_loss = history.history["val_w0_mass_mae_loss"]
w1_mass_mae_loss = history.history["w1_mass_mae_loss"]
val_w1_mass_mae_loss = history.history["val_w1_mass_mae_loss"]

# Create figure with shared axes
fig, axes = plt.subplots(5, 2, figsize=(12, 16), sharex=True, sharey=False)

axes[0, 0].plot(loss, label="Training")
axes[0, 0].plot(val_loss, label="Validation")
axes[0, 0].set_title("Combined Loss")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(False)

axes[0, 1].plot(mae_loss, label="Training")
axes[0, 1].plot(val_mae_loss, label="Validation")
axes[0, 1].set_title("MAE Loss")
axes[0, 1].legend()
axes[0, 1].grid(False)

axes[1, 0].plot(w_mass_mmd0_loss, label="Training")
axes[1, 0].plot(val_w_mass_mmd0_loss, label="Validation")
axes[1, 0].set_title(r"$W^{\ell_0}$ MMD Loss")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()
axes[1, 0].grid(False)

axes[1, 1].plot(w_mass_mmd1_loss, label="Training")
axes[1, 1].plot(val_w_mass_mmd1_loss, label="Validation")
axes[1, 1].set_title(r"$W^{\ell_1}$ MMD Loss")
axes[1, 1].legend()
axes[1, 1].grid(False)

axes[2, 0].plot(w0_mass_mae_loss, label="Training")
axes[2, 0].plot(val_w0_mass_mae_loss, label="Validation")
axes[2, 0].set_title(r"Derived $m_{W^{\ell_0}}$ MAE Loss")
axes[2, 0].set_xlabel("Epochs")
axes[2, 0].legend()
axes[2, 0].grid(False)

axes[2, 1].plot(w1_mass_mae_loss, label="Training")
axes[2, 1].plot(val_w1_mass_mae_loss, label="Validation")
axes[2, 1].set_title(r"Derived $m_{W^{\ell_1}}$ MAE Loss")
axes[2, 1].legend()
axes[2, 1].grid(False)

axes[3, 0].plot(nu_mass_loss, label="Training")
axes[3, 0].plot(val_nu_mass_loss, label="Validation")
axes[3, 0].set_title(r"$m_{\nu}$ Loss")
axes[3, 0].set_ylabel("Loss")
axes[3, 0].legend()
axes[3, 0].grid(False)
axes[3, 0].tick_params(axis='x', rotation=45)

axes[3, 1].plot(neg_r2_loss, label="Training")
axes[3, 1].plot(val_neg_r2_loss, label="Validation")
axes[3, 1].set_title(r"-$R^2$ Loss")
axes[3, 1].legend()
axes[3, 1].grid(False)
axes[3, 1].tick_params(axis='x', rotation=45)

axes[4, 0].plot(higgs_mass_loss, label="Training")
axes[4, 0].plot(val_higgs_mass_loss, label="Validation")
axes[4, 0].set_title(r"$m_{H}$ Loss")
axes[4, 0].set_ylabel("Loss")
axes[4, 0].set_xlabel("Epochs", labelpad=10)
axes[4, 0].tick_params(axis='x', rotation=45)
axes[4, 0].legend()
axes[4, 0].grid(False)

axes[4, 1].plot(dinu_pt_loss, label="Training")
axes[4, 1].plot(val_dinu_pt_loss, label="Validation")
axes[4, 1].set_title(r"$p^{\nu\nu}_{T}$ Loss")
axes[4, 1].set_xlabel("Epochs", labelpad=10)
axes[4, 1].tick_params(axis='x', rotation=45)
axes[4, 1].legend()
axes[4, 1].grid(False)

plt.tight_layout()
plt.savefig(dir_name + name + "_loss.png")
plt.show()
plt.close()

# %%
# save predicted results
np.savez_compressed(dir_name + name + "_data.npz", pred_y=pred_y, test_y=test_y)
sig_pred_inv = np.load(dir_name + name + "_data.npz")["pred_y"]
sig_truth_inv = np.load(dir_name + name + "_data.npz")["test_y"]
print(f"{dir_name + name}_data.npz has been saved.")

# %%
# neutrino mass checking
nu0_4vect = sig_pred_inv[:, :4] - test_x[:, :4]
nu1_4vect = sig_pred_inv[:, 4:8] - test_x[:, 4:8]
nu0_mass_squared = (np.square(nu0_4vect[:, 3]) - np.square(nu0_4vect[:, 0]) - np.square(nu0_4vect[:, 1]) - np.square(nu0_4vect[:, 2]))
nu1_mass_squared = (np.square(nu1_4vect[:, 3]) - np.square(nu1_4vect[:, 0]) - np.square(nu1_4vect[:, 1]) - np.square(nu1_4vect[:, 2]))
plt.hist(nu0_mass_squared, bins=100, histtype="step", label="nu0_mass_squared")
plt.hist(nu1_mass_squared, bins=100, histtype="step", label="nu1_mass_squared")
plt.semilogx()
plt.semilogy()
plt.legend()
plt.show()
plt.close()
print("nu0_mass_squared avg:", np.mean(nu0_mass_squared))
print("nu1_mass_squared avg:", np.mean(nu1_mass_squared))

# %%
lead_time_like = np.square(sig_pred_inv[:, 3]) - np.sum(np.square(sig_pred_inv[:, 0:3]), axis=1)
lead_time_mask = (lead_time_like > 0) * 1
sublead_time_like = np.square(sig_pred_inv[:, 7]) - np.sum(np.square(sig_pred_inv[:, 4:7]), axis=1)
sublead_time_mask = (sublead_time_like > 0) * 1

bin_edges = np.linspace(-5, 1.0e4, 51)
figure = plt.figure(figsize=(10, 8))
plt.hist(
    lead_time_like,
    bins=bin_edges,
    fill=False,
    color="tab:orange",
    histtype="step",
    label=r"$W^{0}$, " + f"{100 * np.sum(lead_time_mask) / len(lead_time_mask):.2f} % TL",
    density=True,
    linewidth=2,
)
plt.hist(
    sublead_time_like,
    bins=bin_edges,
    fill=False,
    color="tab:blue",
    histtype="step",
    label=r"$W^{1}$, " + f"{100 * np.sum(sublead_time_mask) / len(sublead_time_mask):.2f} % TL",
    density=True,
    linewidth=2,
)
plt.hist(
    np.square(w_lead.m),
    bins=bin_edges,
    color="tab:orange",
    fill=True,
    histtype="bar",
    alpha=0.5,
    label=r"$W_{label}^{0}$",
    density=True,
)
plt.hist(
    np.square(w_sublead.m),
    bins=bin_edges,
    color="tab:blue",
    fill=True,
    histtype="bar",
    alpha=0.5,
    label=r"$W_{label}^{1}$",
    density=True,
)
plt.vlines(0, 0, 2e-3, color="red", linestyle="--")
plt.xlim(-2e2, 8e3)
plt.ylim(0, 1.5e-3)
plt.xlabel(r"Derived Mass$^{2}$ [GeV]$^{2}$", labelpad=20)
plt.ylabel("Normalized counts")
plt.legend(loc="best", ncol=1, prop={"size": 20}, columnspacing=1.0, frameon=False, framealpha=0.8)
plt.show()
plt.close()
print(f"m2 with {100 * np.sum(lead_time_mask * sublead_time_mask) / len(lead_time_mask):.2f} % time-like (TL) evts")

# %%
def Bij(particles):
    ROOT.gROOT.SetBatch(True)
    def cglmp(z_xp, z_xn, z_yp, z_yn):
        tr_a = (np.divide(8, np.sqrt(3))) * (z_xp * z_xn + z_yp * z_yn)
        tr_b = 25 * (np.square(z_xp) - np.square(z_yp)) * (np.square(z_xn) - np.square(z_yn))
        tr_c = 100 * (z_xp * z_yp * z_xn * z_yn)
        return tr_a + tr_b + tr_c

    WpBoson = TLorentzVector(*particles[:4])
    WpLepton = TLorentzVector(*particles[4:8])
    WnBoson = TLorentzVector(*particles[8:12])
    WnLepton = TLorentzVector(*particles[12:16])
    Higgs = WpBoson + WnBoson
    Beam_p = TLorentzVector(0, 0, 1, 1)
    Higgsb = Higgs.BoostVector()

    for vec in [WpBoson, WpLepton, WnBoson, WnLepton, Beam_p]:
        vec.Boost(-Higgsb)

    k_per = TVector3(WpBoson.X(), WpBoson.Y(), WpBoson.Z())
    p_per = TVector3(Beam_p.X(), Beam_p.Y(), Beam_p.Z())
    k = k_per.Unit()
    p = p_per.Unit()
    y = p.Dot(k)
    r_length = np.sqrt(1 - y * y)
    r = (1 / r_length) * (p - y * k)
    n = (1 / r_length) * (p.Cross(k))

    WpkBoost = WpBoson.BoostVector()
    WpBoson.Boost(-WpkBoost)
    WpLepton.Boost(-WpkBoost)
    WnkBoost = WnBoson.BoostVector()
    WnBoson.Boost(-WnkBoost)
    WnLepton.Boost(-WnkBoost)

    WpLp = WpLepton.Vect()
    WnLp = WnLepton.Vect()
    WpLp_k = TLorentzVector(WpLp.Dot(n), WpLp.Dot(r), WpLp.Dot(k), WpLepton.E())
    WnLp_k = TLorentzVector(WnLp.Dot(n), WnLp.Dot(r), WnLp.Dot(k), WnLepton.E())

    WpLp_Vect_Mag = WpLp_k.Vect().Mag()
    cos_n_join_p = np.divide(WpLp_k.X(), WpLp_Vect_Mag)
    cos_r_join_p = np.divide(WpLp_k.Y(), WpLp_Vect_Mag)
    cos_k_join_p = np.divide(WpLp_k.Z(), WpLp_Vect_Mag)
    WnLp_Vect_Mag = WnLp_k.Vect().Mag()
    cos_n_join_n = np.divide(WnLp_k.X(), WnLp_Vect_Mag)
    cos_r_join_n = np.divide(WnLp_k.Y(), WnLp_Vect_Mag)
    cos_k_join_n = np.divide(WnLp_k.Z(), WnLp_Vect_Mag)

    B_xy = cglmp(cos_n_join_p, cos_n_join_n, cos_r_join_p, cos_r_join_n)
    B_yz = cglmp(cos_r_join_p, cos_r_join_n, cos_k_join_p, cos_k_join_n)
    B_zx = cglmp(cos_n_join_p, cos_n_join_n, cos_k_join_p, cos_k_join_n)
    return (
        np.array([B_xy, B_yz, B_zx]),
        np.array([cos_n_join_p, cos_r_join_p, cos_k_join_p]),
        np.array([cos_n_join_n, cos_r_join_n, cos_k_join_n]),
    )

def result_generator(particles):
    with multiprocessing.Pool(WORKERS) as pool:
        results = list(pool.map(Bij, particles))
    bij, xi_p, xi_n = zip(*results)
    bij = np.vstack(bij)
    xi_p = np.vstack(xi_p)
    xi_n = np.vstack(xi_n)
    mask = np.any(np.isnan(bij), axis=1)
    return (bij[~mask, :], xi_p[~mask, :], xi_n[~mask, :])

# %%
# reco
particles = np.concatenate(
    [sig_pred_inv[:, 0:4], lead_lep_p4[test_indices], sig_pred_inv[:, 4:8], sublead_lep_p4[test_indices]],
    axis=1,
)
bij = result_generator(particles)[0][0:80_000, :]
xi_p = result_generator(particles)[1][0:80_000, :]
xi_n = result_generator(particles)[2][0:80_000, :]

# truth
particles = np.concatenate(
    [sig_truth_inv[:, 0:4], truth_lead_lep_p4[test_indices], sig_truth_inv[:, 4:8], truth_sublead_lep_p4[test_indices]],
    axis=1,
)
truth_bij = result_generator(particles)[0][0:80_000, :]
truth_xi_p = result_generator(particles)[1][0:80_000, :]
truth_xi_n = result_generator(particles)[2][0:80_000, :]

# %%
w_plot_true = [sig_truth_inv[:, 0], sig_truth_inv[:, 2], sig_truth_inv[:, 3], sig_truth_inv[:, 4], sig_truth_inv[:, 6], sig_truth_inv[:, 7]]
w_plot_pred = [sig_pred_inv[:, 0], sig_pred_inv[:, 2], sig_pred_inv[:, 3], sig_pred_inv[:, 4], sig_pred_inv[:, 6], sig_pred_inv[:, 7]]
w_plot_ranges = [[-200, 200], [-700, 700], [0, 800], [-200, 200], [-600, 600], [0, 800]]
w_plot_labels = [r"$W^{0}\ p_{x}$", r"$W^{0}\ p_{z}$", r"$W^{0}\ E$", r"$W^{1}\ p_{x}$", r"$W^{1}\ p_{z}$", r"$W^{1}\ E$"]
plot.hist_1d_grid(w_plot_true, w_plot_pred, title=w_plot_labels, ranges=w_plot_ranges, xlabel="[GeV]")
plot.hist_2d_grid(w_plot_true, w_plot_pred, title=w_plot_labels, ranges=w_plot_ranges, xlabel="True [GeV]", ylabel="Predicted [GeV]")
plt.close()

# %%
xi_plot_true = [truth_xi_p[:, 0], truth_xi_p[:, 1], truth_xi_p[:, 2], truth_xi_n[:, 0], truth_xi_n[:, 1], truth_xi_n[:, 2]]
xi_plot_pred = [xi_p[:, 0], xi_p[:, 1], xi_p[:, 2], xi_n[:, 0], xi_n[:, 1], xi_n[:, 2]]
xi_plot_ranges = [[-1, 1]] * 6
xi_plot_labels = [r"$\xi^{(0)}_{n}$", r"$\xi^{(0)}_{r}$", r"$\xi^{(0)}_{k}$", r"$\xi^{(1)}_{n}$", r"$\xi^{(1)}_{r}$", r"$\xi^{(1)}_{k}$"]
plot.hist_1d_grid(xi_plot_true, xi_plot_pred, title=xi_plot_labels, ranges=xi_plot_ranges, xlabel="[None]")
plot.hist_2d_grid(xi_plot_true, xi_plot_pred, title=xi_plot_labels, ranges=xi_plot_ranges, xlabel="True [None]", ylabel="Predicted [None]")
plt.close()

# %%
t_end = time.time()
print(f"Time elapsed: {t_end - t_start:.2f} s")
print("Done!")