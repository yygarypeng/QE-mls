# -*- coding: utf-8 -*-
"""W Boson Regressor Module

This module implements a neural network architecture for W boson four-vector
regression using custom loss functions and physics-constrained layers.
"""

import os
import sys

import tensorflow as tf
from keras.saving import register_keras_serializable

# Configuration
HOME_PATH = os.path.abspath("/root/work/QE-mls")
sys.path.append(HOME_PATH + "/qe")

# Callable constants
WORKERS = 16
SEED = 114
GEV = 1e-3
BATCH_SIZE = 512
EPOCHS = 1024
LEARNING_RATE = 1e-4
LOSS_WEIGHTS = {"mae": 1.0, "higgs_mass":1.0 , "w_mass_mmd0": 10, "w_mass_mmd1": 10}

# internal constants
SIGMA_LST = [5.0, 10.0, 50.0, 100.0, 500.0]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress tensorflow information messages


def compute_mmd(x, y, sigma_list=SIGMA_LST):
    """Compute Maximum Mean Discrepancy between two distributions."""
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

    return tf.reduce_mean(XX + YY - 2.0 * XY)


# Loss Functions
def w_mass_loss_fn(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    w0_4Vect, w1_4Vect = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(w0_4Vect[..., 3])
                - tf.reduce_sum(tf.square(w0_4Vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )
    w1_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(w1_4Vect[..., 3])
                - tf.reduce_sum(tf.square(w1_4Vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )

    return tf.reduce_mean(tf.abs(w0_mass - w0_true_mass)), tf.reduce_mean(
        tf.abs(w1_mass - w1_true_mass)
    )


def w_mass_mmd_loss_fn(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    w0_4Vect, w1_4Vect = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(w0_4Vect[..., 3])
                - tf.reduce_sum(tf.square(w0_4Vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )
    w1_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(w1_4Vect[..., 3])
                - tf.reduce_sum(tf.square(w1_4Vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )

    return compute_mmd(w0_mass, w0_true_mass), compute_mmd(w1_mass, w1_true_mass)


def nu_mass_loss_fn(x_batch, y_pred):
    x_batch = tf.cast(x_batch, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    n0_4vect = y_pred[..., :4] - x_batch[..., :4]
    n1_4vect = y_pred[..., 4:8] - x_batch[..., 4:8]

    nu0_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(n0_4vect[..., 3])
                - tf.reduce_sum(tf.square(n0_4vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )
    nu1_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(n1_4vect[..., 3])
                - tf.reduce_sum(tf.square(n1_4vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )

    return tf.reduce_mean(nu0_mass + nu1_mass)


def dinu_pt_loss_fn(x_batch, y_pred):
    x_batch = tf.cast(x_batch, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    n0_4vect = y_pred[..., :4] - x_batch[..., :4]
    n1_4vect = y_pred[..., 4:8] - x_batch[..., 4:8]

    nn_4vect = n0_4vect + n1_4vect
    nn_px_diff = tf.math.maximum(tf.abs(nn_4vect[..., 0] - x_batch[..., 8]), 1e-10)
    nn_py_diff = tf.math.maximum(tf.abs(nn_4vect[..., 1] - x_batch[..., 9]), 1e-10)

    return tf.reduce_mean(nn_px_diff + nn_py_diff)


def higgs_mass_loss_fn(y_pred):
    y_pred = tf.cast(y_pred, tf.float32)

    w0_4Vect, w1_4Vect = y_pred[..., :4], y_pred[..., 4:8]
    higgs_4Vect = w0_4Vect + w1_4Vect

    higgs_mass = tf.sqrt(
        tf.math.maximum(
            tf.abs(
                tf.square(higgs_4Vect[..., 3])
                - tf.reduce_sum(tf.square(higgs_4Vect[..., :3]), axis=-1)
            ),
            1e-10,
        )
    )

    return tf.reduce_mean(tf.abs(higgs_mass - 125.0)) + 1e-10


def mae_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)[:, 0:8]
    y_pred = tf.cast(y_pred, tf.float32)[:, 0:8]
    return tf.reduce_mean(tf.keras.losses.mae(y_true, y_pred))


def neg_r2_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)[:, 0:8]
    y_pred = tf.cast(y_pred, tf.float32)[:, 0:8]
    return (
        tf.reduce_sum(tf.square(y_true - y_pred))
        / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        - 1
    )


@register_keras_serializable(package="ww_regressor")
class CustomModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.current_epoch = tf.Variable(0.0, trainable=False)
        self.metric_dict = {
            "mae_loss": tf.keras.metrics.Mean(name="mae_loss"),
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
            config.pop(key, None)
        if base_model_config is not None:
            base_model = tf.keras.utils.deserialize_keras_object(
                base_model_config, custom_objects=custom_objects
            )
        else:
            raise ValueError("No 'base_model' found in config.")
        return cls(base_model=base_model)

    def get_compile_config(self):
        return {
            "optimizer": tf.keras.utils.serialize_keras_object(self.optimizer),
            "loss_weights": self.loss_weights,
        }

    def compile_from_config(self, config):
            optimizer_config = config["optimizer"]
            optimizer = tf.keras.utils.deserialize_keras_object(
                optimizer_config,
                custom_objects={"Adam": tf.keras.optimizers.Adam}
            )
            # Rebuild optimizer state by recompiling with the base model
            self.compile(
                optimizer=optimizer,
                loss_weights=config.get("loss_weights", None),
            )
            # Ensure optimizer is aware of all trainable variables
            self.optimizer.build(self.trainable_variables)

    def compile(self, optimizer, loss_weights=None, steps_per_execution=1, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        default_weights = {
            "mae": 1.0,
            "nu_mass": 0.0,
            "higgs_mass": 0.0,
            "w0_mass_mae": 0.0,
            "w1_mass_mae": 0.0,
            "w_mass_mmd0": 0.0,
            "w_mass_mmd1": 0.0,
            "dinu_pt": 0.0,
            "neg_r2": 0.0,
        }
        self.loss_weights = {**default_weights, **(loss_weights or {})}
        self.steps_per_execution = steps_per_execution
        # Ensure optimizer is built with all trainable variables
        self.optimizer.build(self.trainable_variables)

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
        total_loss = tf.add_n(
            [self.loss_weights[name] * loss for name, loss in losses.items()]
        )
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


@register_keras_serializable(package="nu_regressor")
class WBosonFourVectorLayer(tf.keras.layers.Layer):
    """Layer to compute W boson four-vectors from leptons and neutrinos."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        lep0, lep1, nu_3mom = inputs
        nu0_3mom, nu1_3mom = nu_3mom[..., :3], nu_3mom[..., 3:]
        nu0_energy = tf.sqrt(
            tf.maximum(
                tf.reduce_sum(tf.square(nu0_3mom), axis=-1, keepdims=True), 1e-10
            )
        )
        nu1_energy = tf.sqrt(
            tf.maximum(
                tf.reduce_sum(tf.square(nu1_3mom), axis=-1, keepdims=True), 1e-10
            )
        )
        nu0_4vect = tf.concat([nu0_3mom, nu0_energy], axis=-1)
        nu1_4vect = tf.concat([nu1_3mom, nu1_energy], axis=-1)
        return tf.concat([lep0 + nu0_4vect, lep1 + nu1_4vect], axis=-1)


def dense_dropout_block(x, units, activation="swish", dropout_rate=0.0, l2=0.0):
    """Dense layer with dropout and batch normalization."""
    x = tf.keras.layers.Dense(
        units,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.L2(l2),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def residual_block(x, units, activation="swish", dropout_rate=0.0, l2=0.0):
    """Residual block with two dense layers."""
    y = dense_dropout_block(x, units, activation, dropout_rate, l2)
    y = dense_dropout_block(y, units, activation, dropout_rate, l2)
    if x.shape[-1] != units:
        x = tf.keras.layers.Dense(units, use_bias=False)(x)
    z = tf.keras.layers.Add()([x, y])
    return tf.keras.layers.Activation(activation)(
        tf.keras.layers.BatchNormalization()(z)
    )


def build_model(input_shape):
    """Build the W boson regressor model."""
    inputs = tf.keras.layers.Input(shape=(input_shape,), dtype=tf.float32)
    x = inputs
    lep0, lep1 = inputs[..., :4], inputs[..., 4:8]

    for _ in range(2):
        x = residual_block(x, 256, dropout_rate=0.3, l2=1e-4)
        x = residual_block(x, 64, dropout_rate=0.3, l2=1e-4)
    for _ in range(2):
        x = residual_block(x, 32, dropout_rate=0.3, l2=1e-4)
        x = residual_block(x, 128, dropout_rate=0.3, l2=1e-4)
    for _ in range(2):
        x = residual_block(x, 128, dropout_rate=0.3, l2=1e-4)
        x = residual_block(x, 32, dropout_rate=0.3, l2=1e-4)
    for _ in range(2):
        x = residual_block(x, 64, dropout_rate=0.3, l2=1e-4)
        x = residual_block(x, 128, dropout_rate=0.3, l2=1e-4)

    x = tf.keras.layers.Dense(16, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)

    nu_3mom = tf.keras.layers.Dense(
        6, activation="linear", kernel_initializer="he_normal"
    )(x)
    
    outputs = WBosonFourVectorLayer()([lep0, lep1, nu_3mom])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class EpochUpdater(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(float(epoch))


class LambdaTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = int(self.model.current_epoch.numpy())
        log_str = f"Epoch {epoch_num}"
        for name, value in logs.items():
            log_str += f"; {name}: {value:.3E}"
        print(log_str)


if __name__ == "__main__":
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Home path: {HOME_PATH}")
