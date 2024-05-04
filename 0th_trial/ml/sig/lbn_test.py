# coding: utf-8

"""
This example demonstrates the usage of the LBN different approaches:
  - TF eager mode
  - TF AutoGraph
  - Keras model / layer
"""

import numpy as np
import tensorflow as tf
from lbn import LBN, LBNLayer


def create_four_vectors(n, p_low=-100.0, p_high=100.0, m_low=0.1, m_high=50.0):
    """
    Creates a numpy array with shape ``n + (4,)`` describing four-vectors of particles whose
    momentum components are uniformly distributed between *p_low* and *p_high*, and masses between
    *m_low* and *m_high*.
    """
    # create random four-vectors
    if not isinstance(n, tuple):
        n = (n,)
    vecs = np.random.uniform(p_low, p_high, n + (4,)).astype(np.float32)

    # the energy is also random and might be lower than the momentum,
    # so draw uniformly distributed masses, and compute and insert the energy
    m = np.abs(np.random.uniform(m_low, m_high, n))
    p = np.sqrt(np.sum(vecs[..., 1:] ** 2, axis=-1))
    E = (p**2 + m**2) ** 0.5
    vecs[..., 0] = E

    return vecs


def test_tf_eager():
    # define 10 random input vectors with batch size 2
    inputs = create_four_vectors((2, 10))

    # initialize the LBN with pair-wise boosting, 10 particles and rest frames
    lbn = LBN(10, boost_mode=LBN.PAIRS)

    # run build, which creates trainable variables and an eager callable in eager mode
    lbn.build(inputs.shape, features=["E", "pt", "eta", "phi", "m", "pair_cos"])

    # show available features
    print(lbn.available_features)

    # build certain features for all 10 boosted particles
    features = lbn(inputs)

    # print features (10 x E, 10 x pt, 10 x eta, 10 x phi, 10 x m, 45 x pair_cos)
    print(features)

    # other members to print:
    # tensors of particle combinations
    #   lbn.particles_E
    #   lbn.particles_px
    #   lbn.particles_py
    #   lbn.particles_pz
    #   lbn.particles_pvec
    #   lbn.particles
    # tensors of rest frame combinations
    #   lbn.restframes_E
    #   lbn.restframes_px
    #   lbn.restframes_py
    #   lbn.restframes_pz
    #   lbn.restframes_pvec
    #   lbn.restframes
    # tensor of boosted particles
    #   lbn.boosted_particles
    # combination weights
    #   lbn.final_particle_weights
    #   lbn.final_restframe_weights


def test_tf_autograph():
    # define 10 random input vectors with batch size 2
    inputs = create_four_vectors((2, 10))

    # initialize the LBN with pair-wise boosting, 10 particles and rest frames
    lbn = LBN(10, boost_mode=LBN.PAIRS)

    # run build, which creates trainable variables and the computational graph in graph mode
    lbn.build(inputs.shape, features=["E", "pt", "eta", "phi", "m", "pair_cos"])

    @tf.function
    def predict(inputs):
        return lbn(inputs)

    # print features
    print(predict(inputs))


def test_keras():
    # define 10 random input vectors with batch size 2
    inputs = create_four_vectors((2, 10))

    # create the LBN layer for an input shape (10, 4), 10 particles and rest frames, pair-wise
    # boosting and a certain set of features to generate
    lbn_layer = LBNLayer(
        (10, 4),
        n_particles=10,
        boost_mode=LBN.PAIRS,
        features=["E", "pt", "eta", "phi", "m", "pair_cos"],
    )

    # define the keras model and add the lbn
    model = tf.keras.models.Sequential()
    model.add(lbn_layer)

    # compile the model (we have to pass a loss or it won't compile)
    model.compile(loss="categorical_crossentropy")

    # compute features
    features = model.predict(inputs)

    print(features)


if __name__ == "__main__":
    print(" LBN eager test ".center(100, "="))
    test_tf_eager()

    print("\n" + " LBN autograph test ".center(100, "="))
    test_tf_autograph()

    print("\n" + " LBN keras test ".center(100, "="))
    test_keras()
