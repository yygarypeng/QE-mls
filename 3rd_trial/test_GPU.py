import sys
sys.path.insert(0, '../qe')
import get_data as gd

import pandas as pd
import numpy as np
import scipy as sp

import os
import glob
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress TensorFlow information messages

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
print(tf.__version__)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # for gpu in gpus:
            # tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

processor = gd.DataProcessor()
path = "/root/data/truth/signal/*npz"
processor.load_files(path)
data = gd.Data(*processor.files)

GEV = 1e3

cut_pre_pt_lead = data.LeadLep["pt"] > -999 * GEV
cut_pre_pt_sub = data.SubLep["pt"] > -999 * GEV
cut_pre_dilep_m = data.diLep["m"] > -999 * GEV
cut_pre_pt_miss = data.MET["pt"] > -999 * GEV

PRE_CUT = cut_pre_pt_lead & cut_pre_pt_sub & cut_pre_dilep_m & cut_pre_pt_miss
del (cut_pre_pt_lead, cut_pre_pt_sub, cut_pre_dilep_m, cut_pre_pt_miss)

BATCH_SIZE = 100000
EPOCHS = 1024
LEARNING_RATE = 5e-4

lead_p = data.LeadLep["m"] == data.LepP["m"]
sublead_p = ~lead_p
lead_m = sublead_p.copy()
sublead_m = lead_p.copy()

w_lead = pd.concat([data.Wp[lead_p], data.Wm[lead_m]], axis=0).sort_index()
w_sublead = pd.concat([data.Wp[sublead_p], data.Wm[sublead_m]], axis=0).sort_index()

int_kin = w_lead[["E", "px", "py", "pz"]][PRE_CUT] / GEV
ROBUST_INT = RobustScaler()
int_kin = ROBUST_INT.fit_transform(int_kin)

met = data.MET[["px", "py"]]
lead_lep = data.LeadLep[["E", "px", "py", "pz"]]
sublead_lep = data.SubLep[["E", "px", "py", "pz"]]

obs_kin = pd.concat([lead_lep, sublead_lep, met], axis=1)[PRE_CUT] / GEV
ROBUST_OBS = RobustScaler()
obs_kin = ROBUST_OBS.fit_transform(obs_kin)

indices_arr = np.arange(int_kin.shape[0], dtype="int")
train_indices, temp_indices = train_test_split(indices_arr.flatten(), train_size=0.6, test_size=0.4, random_state=SEED, shuffle=True)
valid_indices, test_indices = train_test_split(temp_indices, train_size=0.5, test_size=0.5, random_state=SEED)

train_x = obs_kin[train_indices]
test_x = obs_kin[test_indices]
valid_x = obs_kin[valid_indices]
train_y = int_kin[train_indices]
test_y = int_kin[test_indices]
valid_y = int_kin[valid_indices]

print(f"X (Interest)\nTraining data shape: {train_x.shape};\nValidation data shape: {valid_x.shape};\nTesting data shape: {test_x.shape}.")
print(f"Y (Observed)\nTraining data shape: {train_y.shape};\nValidation data shape: {valid_y.shape};\nTesting data shape: {test_y.shape}.")

def preprocess(features, labels):
    return features, labels

def build_input(features, labels, batch_size, num_epochs, is_train=True, buffer_size=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    if is_train:
        dataset = dataset.shuffle(buffer_size=len(features))
    
    dataset = dataset.map(preprocess, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    
    return dataset

train_dataset = build_input(train_x, train_y, BATCH_SIZE, EPOCHS, is_train=True)
valid_dataset = build_input(valid_x, valid_y, BATCH_SIZE, EPOCHS, is_train=False)
test_dataset = build_input(test_x, test_y, BATCH_SIZE, num_epochs=1, is_train=False)

def build_model():
	inputs = tf.keras.layers.Input(shape=(train_x.shape[-1],))
	x = tf.keras.layers.Flatten()(inputs)

	for i in range(5):
		x = tf.keras.layers.Dense(units=128, activation="elu")(x)
		x = tf.keras.layers.Dense(units=128, activation="elu")(x)
		x = tf.keras.layers.Dense(units=32, activation="elu")(x)
		x = tf.keras.layers.Dense(units=32, activation="elu")(x)
		x = tf.keras.layers.Dense(units=8, activation="elu")(x)
		x = tf.keras.layers.Dense(units=8, activation="elu")(x)

	outputs = tf.keras.layers.Dense(units=train_y.shape[-1], activation="linear")(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
	return model

model = build_model()
model.summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=20, mode="auto", baseline=None, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.01)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=EPOCHS,
                    steps_per_epoch=len(train_x) // BATCH_SIZE,
                    validation_steps=len(valid_x) // BATCH_SIZE,
                    callbacks=[stop_early],
                    verbose=2)