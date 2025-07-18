# -*- coding: utf-8 -*-
"""Main script for W Boson Regression Training and Evaluation"""

import time
import os
import glob
import shutil
import numpy as np
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow information messages
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Custom imports
HOME_PATH = os.path.abspath("/root/work/QE-mls")
import sys

sys.path.append(HOME_PATH + "/qe")

# import data
import full_data as data

# Import the W boson regressor module
import nn


def setup_gpu(growth=False):
    """Configure GPU settings for TensorFlow"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], growth)
            print(
                f"{len(gpus)} Physical GPUs, {len(tf.config.list_logical_devices('GPU'))} Logical GPU"
            )
        except RuntimeError as e:
            print(f"GPU setup error: {e}")


def apply_preselection_cuts(lead_lep, sublead_lep, met, *args):
    """Apply preselection cuts to the data"""
    cut_pre_pt_lead = lead_lep.pt > 22
    cut_pre_pt_sub = sublead_lep.pt > 15
    dilep_m = np.sqrt(
        np.square(lead_lep.energy + sublead_lep.energy)
        - np.square(lead_lep.px + sublead_lep.px)
        - np.square(lead_lep.py + sublead_lep.py)
        - np.square(lead_lep.pz + sublead_lep.pz)
    )
    cut_pre_dilep_m = dilep_m > 10
    cut_pre_pt_miss = met.pt > 20
    return cut_pre_pt_lead & cut_pre_pt_sub & cut_pre_dilep_m & cut_pre_pt_miss


def prepare_features(lead_lep, sublead_lep, met, w_lead, w_sublead, pre_cut):
    """Prepare input and target features"""
    # Observed kinematics
    obs_kin = np.column_stack(
        [
            lead_lep.px[pre_cut],
            lead_lep.py[pre_cut],
            lead_lep.pz[pre_cut],
            lead_lep.energy[pre_cut],
            sublead_lep.px[pre_cut],
            sublead_lep.py[pre_cut],
            sublead_lep.pz[pre_cut],
            sublead_lep.energy[pre_cut],
            met.px[pre_cut],
            met.py[pre_cut],
        ]
    )

    # Target kinematics
    int_kin = np.column_stack(
        [
            w_lead.px[pre_cut],
            w_lead.py[pre_cut],
            w_lead.pz[pre_cut],
            w_lead.energy[pre_cut],
            w_sublead.px[pre_cut],
            w_sublead.py[pre_cut],
            w_sublead.pz[pre_cut],
            w_sublead.energy[pre_cut],
            w_lead.m[pre_cut],
            w_sublead.m[pre_cut],
        ]
    )

    return obs_kin, int_kin


def create_datasets(train_x, train_y, valid_x, valid_y):
    """Create TensorFlow datasets"""
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .cache()
        .batch(nn.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        .cache()
        .batch(nn.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset, valid_dataset


def plot_training_history(history, dir_name, name):
    """Plot and save training history"""
    metrics = [
        ("loss", "val_loss", "Combined Loss"),
        ("mae_loss", "val_mae_loss", "MAE Loss"),
        ("w_mass_mmd0_loss", "val_w_mass_mmd0_loss", r"$W_{0}$ MMD Loss"),
        ("w_mass_mmd1_loss", "val_w_mass_mmd1_loss", r"$W_{1}$ MMD Loss"),
        ("w0_mass_mae_loss", "val_w0_mass_mae_loss", r"Derived $m_{W_{0}}$ MAE Loss"),
        ( "w1_mass_mae_loss", "val_w1_mass_mae_loss", r"Derived $m_{W_{1}}$ MAE Loss"),
        ("nu_mass_loss", "val_nu_mass_loss", r"$m_{\nu}$ Loss"),
        ("neg_r2_loss", "val_neg_r2_loss", r"-$R^2$ Loss"),
        ("higgs_mass_loss", "val_higgs_mass_loss", r"$m_{H}$ Loss"),
        ("dinu_pt_loss", "val_dinu_pt_loss", r"$p^{\nu\nu}_{T}$ Loss"),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(12, 16), dpi=500, sharex=True)
    for idx, (train_key, val_key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(history.history[train_key], label="Training")
        ax.plot(history.history[val_key], label="Validation")
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
        if idx // 2 >= 3:
            ax.tick_params(axis="x", rotation=45)
        if idx % 2 == 0:
            ax.set_ylabel("Loss")
        if idx // 2 == 4:
            ax.set_xlabel("Epochs", labelpad=10)

    plt.tight_layout()
    plt.savefig(f"{dir_name}{name}_loss.png", dpi=500)
    plt.close()


def main():
    t_start = time.time()
    
    # Setup environment
    setup_gpu()

    # Load data
    lead_lep, sublead_lep, met = data.Lead_lep(), data.Sublead_lep(), data.Met()
    w_lead, w_sublead = data.Lead_w(), data.Sublead_w()

    # Apply preselection
    pre_cut = apply_preselection_cuts(lead_lep, sublead_lep, met, w_lead, w_sublead)
    print(f"Events after preselection: {np.sum(pre_cut)}")

    # Prepare features
    obs_kin, int_kin = prepare_features(lead_lep, sublead_lep, met, w_lead, w_sublead, pre_cut)

    # Train-test split
    indices = np.arange(len(obs_kin))
    train_idx, temp_idx = train_test_split(
        indices, train_size=0.8, random_state=nn.SEED
    )
    valid_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, random_state=nn.SEED
    )

    train_x, valid_x, test_x = obs_kin[train_idx], obs_kin[valid_idx], obs_kin[test_idx]
    train_y, valid_y, _ = int_kin[train_idx], int_kin[valid_idx], int_kin[test_idx]

    print(
        f"Training: {train_x.shape}, Validation: {valid_x.shape}, Testing: {test_x.shape}"
    )

    # Create datasets
    train_dataset, valid_dataset = create_datasets(
        tf.cast(train_x, tf.float32),
        tf.cast(train_y, tf.float32),
        tf.cast(valid_x, tf.float32),
        tf.cast(valid_y, tf.float32),
    )

    # Build and train model
    base_model = nn.build_model(input_shape=train_x.shape[-1])
    model = nn.CustomModel(base_model)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=nn.LEARNING_RATE),
        loss_weights=nn.LOSS_WEIGHTS,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        nn.EpochUpdater(),
        nn.LambdaTracker(),
    ]

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=nn.EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    # Save model
    dir_name = f"{HOME_PATH}/13th_trial/reg_main/ww_resregressor_result/"
    name = "ww_resregressor"
    savedmodel_path = f"{dir_name}saved_model"

    if os.path.exists(dir_name):
        for item in glob.glob(f"{dir_name}*{name}*"):
            shutil.rmtree(item) if os.path.isdir(item) else os.remove(item)
    else:
        os.makedirs(dir_name)

    model.save(f"{dir_name}{name}.keras", overwrite=True)
    tf.saved_model.save(model, savedmodel_path)

    # Predict results
    pred_y = model.predict(test_x)

    # Plot training history
    plot_training_history(history, dir_name, name)

    # Neutrino mass check
    nu0_4vect = pred_y[:, :4] - test_x[:, :4]
    nu1_4vect = pred_y[:, 4:8] - test_x[:, 4:8]
    nu0_mass_sq = nu0_4vect[:, 3] ** 2 - np.sum(nu0_4vect[:, :3] ** 2, axis=1)
    nu1_mass_sq = nu1_4vect[:, 3] ** 2 - np.sum(nu1_4vect[:, :3] ** 2, axis=1)
    print(f"nu0_mass_squared avg: {np.mean(nu0_mass_sq):.2f}")
    print(f"nu1_mass_squared avg: {np.mean(nu1_mass_sq):.2f}")

    # Time-like check
    lead_time_like = pred_y[:, 3] ** 2 - np.sum(pred_y[:, :3] ** 2, axis=1)
    sublead_time_like = pred_y[:, 7] ** 2 - np.sum(pred_y[:, 4:7] ** 2, axis=1)
    time_like_pct = (
        100 * np.sum((lead_time_like > 0) & (sublead_time_like > 0)) / len(pred_y)
    )
    print(f"Time-like events: {time_like_pct:.2f}%")

    print(f"Time elapsed: {time.time() - t_start:.2f} s")
    print("Done!")


if __name__ == "__main__":
    main()
