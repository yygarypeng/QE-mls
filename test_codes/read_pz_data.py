import pandas as pd
import numpy as np
import scipy as sp
import gc
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress tensorflow imformation messages

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf

import get_data as gd
import atlas_plot as ap

plot = ap.Plotter()

print(tf.__version__)
print(tf.config.list_physical_devices())
print()

seed = 42  # set random seed
sampling = int(1e6)

np.random.seed(seed)
processor = gd.DataProcessor(sampling=sampling, processor=30)
path = "/root/data/truth/signal/*npz"
processor.load_files(path)
data = gd.Data(*processor.files)

GEV = 1e3

cut_pre_pt_lead = data.LeadLep["pt"] > 22 * GEV
cut_pre_pt_sub = data.SubLep["pt"] > 15 * GEV
cut_pre_dilep_m = data.diLep["m"] > 10 * GEV
cut_pre_pt_miss = data.MET["pt"] > 20 * GEV
cut_pre = cut_pre_pt_lead & cut_pre_pt_sub & cut_pre_dilep_m & cut_pre_pt_miss

del (cut_pre_pt_lead, cut_pre_pt_sub, cut_pre_dilep_m, cut_pre_pt_miss)

# inputs -> observed params
lep_p = processor.process_part(data.LepP)
lep_m = processor.process_part(data.LepM)
lep_kin = pd.concat([lep_p.iloc[:, :4], lep_m.iloc[:, :4]], axis=1)
MET = processor.process_MET(data.MET)
dilep_kin = processor.process_dipart(data.LepP, data.LepM)
obs_kin = pd.concat([MET.iloc[:, 1:3], lep_kin], axis=1)[cut_pre]
print("obs_kin shape:", obs_kin.shape)
print(obs_kin.head(3))
print()

# targets -> interested unknowns
int_kin = pd.DataFrame(processor.process_dipart(data.NuP, data.NuM)[["pz"]])[cut_pre]
print("int_kin shape:", int_kin.shape)
print(int_kin.head(3))
print()

del (processor, lep_p, lep_m, MET, dilep_kin)
gc.collect()

SCALAR_int_ru = RobustScaler()
int_kin = SCALAR_int_ru.fit_transform(int_kin)
SCALAR_int_mm = MinMaxScaler()
int_kin = SCALAR_int_mm.fit_transform(int_kin)

SCALAR_obs_ru = RobustScaler()
obs_kin = SCALAR_obs_ru.fit_transform(obs_kin)
SCALAR_obs_mm = MinMaxScaler()
obs_kin = SCALAR_obs_mm.fit_transform(obs_kin)

indices_arr = np.arange(int_kin.shape[0], dtype="int")
train_indices, temp_indices = train_test_split(
    indices_arr.flatten(),
    train_size=0.4,
    test_size=0.6,
    random_state=seed,
    shuffle=True,
)
valid_indices, test_indices = train_test_split(
    temp_indices, train_size=0.5, test_size=0.5, random_state=42
)

train_x = obs_kin[train_indices]
test_x = obs_kin[test_indices]
valid_x = obs_kin[valid_indices]
train_y = int_kin[train_indices]
test_y = int_kin[test_indices]
valid_y = int_kin[valid_indices]

print(
    f"X (Interest)\nTraining data shape: {train_x.shape};\nValiding data shape: {valid_x.shape};\nTesting data shape: {test_x.shape}."
)
print(
    f"Y (Observed)\nTraining data shape: {train_y.shape};\nValiding data shape: {valid_y.shape};\nTesting data shape: {test_y.shape}."
)
print()

model = tf.keras.models.load_model("./DNN_pz.h5")

# predict
pred_y = model.predict(test_x)
sig_pred = pred_y
sig_truth = test_y

print(f"Truth mean: {np.mean(sig_truth[:,0]):.3f}, std: {np.std(sig_truth[:,0]):.3f}")

np.savez_compressed("sig_pz.npz", pred=sig_pred[:, 0], truth=sig_truth[:, 0])

print("---- Finish signal output ----")

# load bkg data
np.random.seed(seed)
processor = gd.DataProcessor(sampling=sampling, processor=30)
path = "/root/data/truth/background/*npz"
processor.load_files(path)
data = gd.Data(*processor.files)

cut_pre_pt_lead = data.LeadLep["pt"] > 22 * GEV
cut_pre_pt_sub = data.SubLep["pt"] > 15 * GEV
cut_pre_dilep_m = data.diLep["m"] > 10 * GEV
cut_pre_pt_miss = data.MET["pt"] > 20 * GEV
cut_pre = cut_pre_pt_lead & cut_pre_pt_sub & cut_pre_dilep_m & cut_pre_pt_miss

del (cut_pre_pt_lead, cut_pre_pt_sub, cut_pre_dilep_m, cut_pre_pt_miss)

# inputs -> observed params
lep_p = processor.process_part(data.LepP)
lep_m = processor.process_part(data.LepM)
lep_kin = pd.concat([lep_p.iloc[:, :4], lep_m.iloc[:, :4]], axis=1)
MET = processor.process_MET(data.MET)
dilep_kin = processor.process_dipart(data.LepP, data.LepM)
bkg_obs_kin = pd.concat([MET.iloc[:, 1:3], lep_kin], axis=1)[cut_pre]
print("bkg_obs_kin shape:", bkg_obs_kin.shape)
print(bkg_obs_kin.head(3))
print()

# targets -> interested unknowns
bkg_int_kin = pd.DataFrame(processor.process_dipart(data.NuP, data.NuM)[["pz"]])[
    cut_pre
]
print("bkg_int_kin shape:", bkg_int_kin.shape)
print(bkg_int_kin.head(3))
print()

SCALAR_int_ru = RobustScaler()
bkg_int_kin = SCALAR_int_ru.fit_transform(bkg_int_kin)
SCALAR_int_mm = MinMaxScaler()
bkg_int_kin = SCALAR_int_mm.fit_transform(bkg_int_kin)

SCALAR_obs_ru = RobustScaler()
bkg_obs_kin = SCALAR_obs_ru.fit_transform(bkg_obs_kin)
SCALAR_obs_mm = MinMaxScaler()
bkg_obs_kin = SCALAR_obs_mm.fit_transform(bkg_obs_kin)

# predict
pred_y = model.predict(bkg_obs_kin)
bkg_pred = pred_y
bkg_truth = bkg_int_kin

print(f"Truth mean: {np.mean(bkg_truth[:,0]):.3f}, std: {np.std(bkg_truth[:,0]):.3f}")

np.savez_compressed("bkg_pz.npz", pred=bkg_pred[:, 0], truth=bkg_truth[:, 0])

print("---- Finish background output ----")

sig = np.load("./sig_pz.npz")
sig_truth, sig_pred = sig["truth"], sig["pred"]
range = [0.43, 0.57]

plot.plot_hist(
    [sig_truth, sig_pred],
    [r"$p_{z\ truth}^{\nu\nu}$", r"$p_{z\ pred}^{\nu\nu}$"],
    r"Sig. norm. $p^{\nu\nu}_{z}$"
    + f" with RMSE: {np.sqrt(mean_squared_error(sig_truth, sig_pred)):.3f}",
    range=range,
    xlabel=r"Norm. $p_{z}$ [unit]",
)

plot.plot_2d_histogram(
    sig_truth,
    sig_pred,
    r"Sig. $p_{z}^{\nu\nu}$ "
    + f"with Pearson coeff: {sp.stats.pearsonr(sig_truth, sig_pred)[0]:.3f}",
    range=range,
)

bkg = np.load("./bkg_pz.npz")
bkg_truth, bkg_pred = bkg["truth"], bkg["pred"]

range = [0.3, 0.7]

plot.plot_hist(
    [bkg_truth, bkg_pred],
    [r"$p_{z\ truth}^{\nu\nu}$", r"$p_{z\ pred}^{\nu\nu}$"],
    r"Bkg. norm. $p^{\nu\nu}_{z}$"
    + f" with RMSE: {np.sqrt(mean_squared_error(bkg_truth, bkg_pred)):.3f}",
    range=range,
    xlabel=r"Norm. $p_{z}$ [unit]",
)

plot.plot_2d_histogram(
    bkg_truth,
    bkg_pred,
    r"Bkg. $p_{z}^{\nu\nu}$ "
    + f"with Pearson coeff: {sp.stats.pearsonr(bkg_truth, bkg_pred)[0]:.3f}",
    range=range,
)
