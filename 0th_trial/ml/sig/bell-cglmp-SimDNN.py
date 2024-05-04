import gc
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_files_names(path):
    files_name = glob.glob(path)
    return files_name


path = "./truth/*/*.npz"
files_name = get_files_names(path)
print(files_name)


def get_data(path):
    try:
        with np.load(path, allow_pickle=True) as f:
            data_dict = {name: f[name] for name in f.files}
            return pd.DataFrame(data_dict)
    except FileNotFoundError:
        print("File not found!")
        return pd.DataFrame()


files = []
files_name.sort()
for f in files_name:
    files.append(get_data(f))

# need to check the order of data name.
print(files_name)

# need to follow the order of data name.
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
) = files
GEV = 1e3

del files_name, files
gc.collect()

# Some constants
GEV = 1e3
# RMV_EVT = [638488, 835579, 2168342] # escape some mathmetical errors.
RMV_EVT = []  # escape some mathmetical errors.

# Kinemetic info of leptons.
lep_kin = (
    pd.DataFrame(
        {
            "lep_p_E": LepP["E"],
            "lep_p_px": LepP["px"],
            "lep_p_py": LepP["py"],
            "lep_p_pz": LepP["pz"],
            "lep_m_E": LepM["E"],
            "lep_m_px": LepM["px"],
            "lep_m_py": LepM["py"],
            "lep_m_pz": LepM["pz"],
        }
    )
    / GEV
)

# check format l+ -> (E, px, py, pz); then, append l- with the same format of l+.
print(lep_kin.shape)
lep_kin.drop(RMV_EVT, inplace=True)
print(lep_kin.shape)
lep_kin.head(5)


# Kinemetic info of neutirnos.
nu_kin = (
    pd.DataFrame(
        {
            "nu_p_E": NuP["E"],
            "nu_p_px": NuP["px"],
            "nu_p_py": NuP["py"],
            "nu_p_pz": NuP["pz"],
            "nu_m_E": NuM["E"],
            "nu_m_px": NuM["px"],
            "nu_m_py": NuM["py"],
            "nu_m_pz": NuM["pz"],
        }
    )
    / GEV
)

# check format nu+ -> (E, px, py, pz); then, append nu- with the same format of l+.
print(nu_kin.shape)
nu_kin.drop(RMV_EVT, inplace=True)
print(nu_kin.shape)
nu_kin.head(5)


# CGLMP.
CGLMP = pd.DataFrame(
    {
        "Bxy": CGLMP["Bxy"],
        "Byz": CGLMP["Byz"],
        "Bzx": CGLMP["Bzx"],
    }
)

# check
print(CGLMP.shape)
CGLMP.drop(RMV_EVT, inplace=True)
print(CGLMP.shape)
CGLMP.head(5)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

print(tf.config.list_physical_devices())
from tensorflow import keras
from tensorflow.keras import layers

# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
print(tf.__version__)

# install lbn
from lbn import LBN, LBNLayer

from sklearn.preprocessing import StandardScaler

SCALAR = StandardScaler()
Bxy_std = SCALAR.fit_transform(
    [[x] for x in CGLMP["Bxy"]]
).flatten()  # zero mean and unit variance
plt.hist(Bxy_std, bins=50)
plt.title("Before reweighting Bxy_std")
plt.show()
plt.savefig("before_reweighting_Bxy_std.png")
plt.close()
print(Bxy_std.max(), Bxy_std.min())
print("std:", Bxy_std.std())
print(len(Bxy_std))

# Undersampling to an uniform distribution
n = int(1e3)
step = (Bxy_std.max() - Bxy_std.min()) / n
intervals = [Bxy_std.min() + i * step for i in range(n)]

sampling = int(1e3)
indices_arr = np.empty((n - 1, sampling), dtype=int)
for i in range(n - 1):
    try:
        indices_arr[i] = np.random.choice(
            np.where((intervals[i] < Bxy_std) * (Bxy_std <= intervals[i + 1]))[0],
            size=sampling,
            replace=True,
        )
    except ValueError:
        print("Cannot take a larger sample than population when 'replace=False")
        break

from sklearn.model_selection import train_test_split

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
    "nu_p_E",
    "nu_p_px",
    "nu_p_py",
    "nu_p_pz",
    "nu_m_E",
    "nu_m_px",
    "nu_m_py",
    "nu_m_pz",
]


def reshape_features(inputs, features):
    outputs = np.stack([inputs[features[0:4]], inputs[features[4:8]]], axis=1)
    return outputs


train_indices, temp_indices = train_test_split(
    indices_arr.flatten(), train_size=0.8, test_size=0.2, random_state=42
)
valid_indices, test_indices = train_test_split(
    temp_indices, train_size=0.5, test_size=0.5, random_state=42
)

lep_train = reshape_features(lep_kin.iloc[train_indices], lepton_features)
lep_valid = reshape_features(lep_kin.iloc[valid_indices], lepton_features)
lep_test = reshape_features(lep_kin.iloc[test_indices], lepton_features)

nu_train = reshape_features(nu_kin.iloc[train_indices], neutrino_features)
nu_valid = reshape_features(nu_kin.iloc[valid_indices], neutrino_features)
nu_test = reshape_features(nu_kin.iloc[test_indices], neutrino_features)

Bxy_train = Bxy_std[train_indices]
Bxy_valid = Bxy_std[valid_indices]
Bxy_test = Bxy_std[test_indices]
plt.hist(Bxy_train, bins=50)
plt.title("Train (Bxy)")
plt.show()
plt.savefig("train_Bxy.png")
plt.close()
plt.hist(Bxy_valid, bins=50)
plt.title("Valid (Bxy)")
plt.show()
plt.savefig("valid_Bxy.png")
plt.close()
plt.hist(Bxy_test, bins=50)
plt.title("Test (Bxy)")
plt.show()
plt.savefig("test_Bxy.png")
plt.close()


def stack_parts(input1, input2):
    outputs = np.concatenate([input1, input2], axis=1)
    return outputs


if (
    lep_train.shape == nu_train.shape
    and lep_test.shape == nu_test.shape
    and lep_valid.shape == nu_valid.shape
):
    print("With the same shapes...\n")
    # print(f'Training data shape: {lep_train.shape}\nTesting data shape: {lep_test.shape}\nValidation data shape: {lep_valid.shape}')
    train = stack_parts(lep_train, nu_train)
    valid = stack_parts(lep_valid, nu_valid)
    test = stack_parts(lep_test, nu_test)

    print(
        print(
            f"Training data shape: {train.shape}\nTesting data shape: {test.shape}\nValidation data shape: {valid.shape}"
        )
    )

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

else:
    print("The shape of leptons are NOT the same with neutrinos shape...\n")
    print(
        f"Training data shape: {lep_train.shape}, {nu_train.shape}\nTesting data shape: {lep_test.shape}, {nu_test.shape}\nValidation data shape: {lep_valid.shape}, {nu_valid.shape}"
    )


# Create random data for 50 figures (4x4 matrices)
num_figures = 30
figure_size = 4

# Calculate the grid dimensions (rows x columns)
num_rows = 5
num_cols = 6

# Create a subplot grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the figures and plot them on subplots
for i, ax in enumerate(axes):
    ax.imshow(
        train[i].T,
        cmap="viridis",
        origin="lower",
        vmin=train[0 : num_figures + 1].min(),
        vmax=train[0 : num_figures + 1].max(),
    )
    ax.set_title(f"{Bxy_train[i]:2f}")
    ax.axis("off")  # Turn off axis labels
plt.tight_layout()
plt.show()
plt.savefig("train_data.png")
plt.close()

# define model
model = keras.models.Sequential()

# LBN layer
input_shape = (4, 4)  # 4 particles, 16 vector components
N_combinations = 16  # number of composite particles/rest frames
model.add(LBNLayer(input_shape, 
                   N_combinations, 
                   boost_mode=LBN.PAIRS,
                   features=["E", "pt", "eta", "phi", "m", "pair_cos"],))

# Simple DNN hidden layers
model.add(layers.Dense(units=64, activation="elu"))
model.add(layers.Dense(units=64, activation="elu"))

# Last dense layer
model.add(layers.Dense(units=1, activation="linear"))

# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3), loss="mse")
model.summary()

# Fit the model
stop_early = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=10,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
history = model.fit(
    x=train,
    y=Bxy_train,
    validation_data=(valid, Bxy_valid),
    epochs=100,
    batch_size=256,
    callbacks=stop_early,
)
print("\nFinish!")

Bxy_pred = model.predict(test)

from sklearn.metrics import mean_squared_error

print(
    f"""RMSE: {mean_squared_error(Bxy_test, Bxy_pred)}
MAX of pred: {Bxy_pred.max()}; MIN of pred: {Bxy_pred.min()}
MAX of test: {Bxy_test.max()}; MIN of test: {Bxy_test.min()}"""
)

# Plot the results
fig = plt.figure(figsize=(5, 5), dpi=120)
ax = fig.add_subplot()
plt.plot(Bxy_pred, Bxy_test, ".", color="tab:blue", alpha=0.2, markersize=3)
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.xlabel("pred")
plt.ylabel("true")
ax.set_aspect("equal", adjustable="box")
plt.show()
plt.savefig("pred_vs_true.png")
plt.close()

fig = plt.figure(figsize=(8, 5), dpi=120)
plt.plot(history.history["loss"], lw=2.5, label="Train", alpha=0.8)
plt.plot(history.history["val_loss"], lw=2.5, label="Validation", alpha=0.8)
plt.semilogy()
plt.title("Epoch vs MSE")
plt.xlabel("epoch")
plt.ylabel("Loss (MSE)")
plt.legend(loc="best")
plt.show()
plt.savefig("epoch_vs_mse.png")
plt.close()

plt.hist(CGLMP["Bxy"], bins=50)
plt.title("Original Bxy")
print(CGLMP["Bxy"].max(), CGLMP["Bxy"].min())
print("std:", CGLMP["Bxy"].std())
print("mean:", CGLMP["Bxy"].mean())
print(len(CGLMP["Bxy"]))
plt.show()
plt.savefig("original_Bxy.png")
plt.close()

plt.hist(Higgs["m"], bins=50)
plt.title("Original Higgs mass")
plt.show()
plt.savefig("original_Higgs_mass.png")
plt.close()
print(Higgs["m"].max(), (Higgs["m"].min()))
print("std:", (Higgs["m"].std()))
print("mean:", (Higgs["m"].mean()))
print(len((Higgs["m"])))
