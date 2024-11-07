import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import full_data

## prepare inputs
# reco
lead_lep = full_data.Lead_lep()
sublead_lep = full_data.Sublead_lep()
dilep = full_data.Dilep()
met = full_data.Met()

# truth
truth_lead_lep = full_data.Truth_lead_lep()
truth_sublead_lep = full_data.Truth_sublead_lep()
Truth_dilep = full_data.Truth_dilep()
truth_met = full_data.Truth_met()

# target Ws
w_lead = full_data.Lead_w()
w_sublead = full_data.Sublead_w()

## Interesting variables
w_lead_px = pd.DataFrame(w_lead.px)
w_lead_py = pd.DataFrame(w_lead.py)
w_lead_pz = pd.DataFrame(w_lead.pz)
w_lead_e = pd.DataFrame(w_lead.energy)
w_lead_log_e = pd.DataFrame(np.log(w_lead_e))
w_lead_sqrt_e = pd.DataFrame(np.sqrt(w_lead_e))
w_sublead_px = pd.DataFrame(w_sublead.px)
w_sublead_py = pd.DataFrame(w_sublead.py)
w_sublead_pz = pd.DataFrame(w_sublead.pz)
w_sublead_e = pd.DataFrame(w_sublead.energy)
w_sublead_log_e = pd.DataFrame(np.log(w_sublead_e))
w_sublead_sqrt_e = pd.DataFrame(np.sqrt(w_sublead_e))
w_lead_m = pd.DataFrame(w_lead.m)
w_sublead_m = pd.DataFrame(w_sublead.m)

# Kinematics of interesting variables (target for training)
int_kin = np.concatenate(
    [
        w_lead_px,
        w_lead_py,
        w_lead_pz,
        w_lead_e,
        w_sublead_px,
        w_sublead_py,
        w_sublead_pz,
        w_sublead_e,
    ],
    axis=-1,
)[0:100_000]
print("int_kin shape:", int_kin.shape)
# int_kin = int_kin.to_numpy()  # convert to numpy array
print(type(int_kin))

## Observing variables
obs_kin = np.column_stack(
    (
        lead_lep.px,
        lead_lep.py,
        np.log(lead_lep.energy),
        lead_lep.eta,
        sublead_lep.px,
        sublead_lep.py,
        np.log(sublead_lep.energy),
        sublead_lep.eta,
        met.px,
        met.py,
    )
)[0:100_000]

# Kinematics of observing variables (inputs for training)
print("int_kin shape:", obs_kin.shape)
# print(print(obs_kin.describe()))
# obs_kin = obs_kin.to_numpy() # convert to numpy array

ROBUST_OBS = StandardScaler()
obs_kin = ROBUST_OBS.fit_transform(obs_kin)
print(type(obs_kin))


## Test onnx model 
model_path = '/root/work/QE-mls/8th_trial/ww_resregressor_result/'
inputs = obs_kin.astype(np.float32)
# ! GPU broken
# sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
sess = ort.InferenceSession(model_path + "ww_resregressor.onnx")
results_ort = sess.run(None, {"inputs": inputs})

import tensorflow as tf

model = tf.saved_model.load(model_path + "saved_model")
infer = model.signatures["serving_default"] # Access the serving function
tf_inputs = tf.convert_to_tensor(inputs.copy())
results_tf = infer(tf_inputs)

# Print the results
print("TensorFlow model results:")
print(results_tf)
print("ONNX model results:")
print(results_ort)

# Compare the results
for ort_res, tf_res in zip(results_ort, results_tf.values()):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-3, atol=1e-1)
    
print("Results match")