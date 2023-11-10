# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flow import *
from utils import *
import get_data as gd
import gc

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from lbn import LBN, LBNLayer

print(tf.__version__)
print(tf.config.list_physical_devices())
print()

# %% [markdown]
# Load data

# %%
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
# print("lep_kin shape:", lep_kin.shape)
# print(lep_kin.head(5), end="\n")

# observed (Y)
MET_kin = processor.process_MET(MET).iloc[:, 1:3]
MET_kin = pd.concat([MET_kin, lep_kin], axis=1)
print("MET_kin shape:", MET_kin.shape)
print(MET_kin.head(3))

# interest (X)
dinu_kin = processor.process_dinu(NuP, NuM)
# print("dinu_kin shape:", dinu_kin.shape)
# print(dinu_kin.head(5), end="\n")

dinu_kin = pd.concat([dinu_kin, lep_kin], axis=1)
print("dinu_kin shape:", dinu_kin.shape)
print(dinu_kin.head(3))
print()

del processor  # Clear the instance
del (
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
)  # Clear the dataframes
del lep_kin
gc.collect()

# %% [markdown]
# Preprocessing

# %%
# normalize
from sklearn.preprocessing import StandardScaler

SCALAR_int = StandardScaler()
norm_var = SCALAR_int.fit_transform(dinu_kin)
dinu_kin = norm_var

SCALAR_MET = StandardScaler()
norm_var = SCALAR_MET.fit_transform(MET_kin)
MET_kin = norm_var

del norm_var

# %%
from sklearn.model_selection import train_test_split

np.random.seed(42)  # set random seed
indices_arr = np.arange(dinu_kin.shape[0], dtype="int")
indices_arr = np.random.choice(indices_arr, int(1e4))
train_indices, test_indices = train_test_split(
    indices_arr.flatten(), train_size=0.8, test_size=0.2, random_state=42
)

train_x = dinu_kin[train_indices]
test_x = dinu_kin[test_indices]
train_y = MET_kin[train_indices]
test_y = MET_kin[test_indices]

print(
    f"X (Interest)\nTraining data shape: {train_x.shape};\nTesting data shape: {test_x.shape}."
)
print(
    f"Y (Observed)\nTraining data shape: {train_y.shape};\nTesting data shape: {test_y.shape}."
)
print()


# %% [markdown]
# Setup
def build_model(lbn=False):
    # define model
    model = keras.models.Sequential()

    # use LBN layer or not
    input_shape = (3, 4)  # 3 particles with 4 vector
    if lbn == True:
        N_combinations = 3  # number of composite particles/rest frames
        model.add(
            LBNLayer(
                input_shape,
                N_combinations,
                boost_mode=LBN.PAIRS,
                # features=['E', 'beta', 'eta', 'gamma', 'm', 'p', 'pair_cos', 'pair_dr', 'pair_ds', 'pair_dy', 'phi', 'pt', 'px', 'py', 'pz']
                features=["px", "py", "pz", "E", "pt", "eta", "phi", "m"],
            )
        )
        print("Use LBN layer...")
    else:
        model.add(layers.Flatten())
        print("Not using LBN layer...")

    # Simple DNN hidden layers
    model.add(layers.Dense(units=32, activation="elu"))
    model.add(layers.Dense(units=32, activation="elu"))
    model.add(layers.Dense(units=32, activation="elu"))

    # Last dense layer
    model.add(layers.Dense(units=1, activation="linear"))

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    model.summary()

    return model


model = build_model(lbn=False)

# Fit the model
history = model.fit(
    x=train_x,
    y=train_y,
    epochs=32,
    batch_size=512,
    verbose=2,
)


# %%
x_pred = model.predict(test_y)
pz_pred = x_pred[:, 3]
pt_pred = np.sqrt(np.square(x_pred[:, 1]) + np.square(x_pred[:, 2]))
E_pred = x_pred[:, 0]
pz_truth = test_x[:, 3]
pt_truth = np.sqrt(np.square(test_x[:, 1]) + np.square(test_x[:, 2]))
E_truth = test_x[:, 0]


# %%
def plot_loss_history(history):
    fig = plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(history.history["loss"], lw=2.5, label="Train", alpha=0.8)
    plt.plot(history.history["val_loss"], lw=2.5, label="Validation", alpha=0.8)
    plt.semilogy()
    plt.title("Epoch vs MSE")
    plt.xlabel("epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend(loc="best")
    plt.savefig("DNN_loss.png")
    plt.show()
    plt.close()


def plot_2d_histogram(pred, truth, title, save_name, bins=100):
    hist, xedges, yedges = np.histogram2d(pred.flatten(), truth, bins=(bins, bins))
    hist = np.ma.masked_where(hist == 0, hist)
    fig = plt.figure(figsize=(6, 6), dpi=120)
    plt.pcolormesh(xedges, yedges, hist.T, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("Frequency")
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.savefig(save_name)
    plt.axis("equal")
    plt.show()
    plt.close()


# %%
# Plot the results
import scipy as sp

print(f"pz -> Pearson coeff: {sp.stats.pearsonr(pz_truth, pz_pred)[0]:.3f}")
print(f"E  -> Pearson coeff: {sp.stats.pearsonr(E_truth, E_pred)[0]:.3f}")
print(f"pt -> Pearson coeff: {sp.stats.pearsonr(pt_truth, pt_pred)[0]:.3f}")
plot_2d_histogram(pz_truth, pz_pred, r"$p^{\nu\nu}_{z}$", save_name="DNN_pz.png")
plot_2d_histogram(E_truth, E_pred, r"$E^{\nu\nu}$", save_name="DNN_energy.png")
plot_2d_histogram(pt_truth, pt_pred, r"$p^{\nu\nu}_{T}$", save_name="DNN_pt.png")

print("====================== Finished!! ======================")
