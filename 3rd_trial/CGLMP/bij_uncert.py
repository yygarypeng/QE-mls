import time

t_start = time.time()


import os
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import ROOT
from ROOT import TLorentzVector, TVector3
import multiprocessing


# some constant
GEV = 1e-3
WORKERS = 16
RNG = np.random.default_rng(2024)
path = "./full_345324_data.h5"


reco_e = pd.read_hdf(path, "reco_e")
reco_leadlep = pd.read_hdf(path, "reco_leadlep")
reco_m = pd.read_hdf(path, "reco_m")
reco_met = pd.read_hdf(path, "reco_met")
reco_subleadlep = pd.read_hdf(path, "reco_subleadlep")
truth_Bij = pd.read_hdf(path, "truth_Bij")
truth_e = pd.read_hdf(path, "truth_e")
truth_h = pd.read_hdf(path, "truth_h")
truth_leadlep = pd.read_hdf(path, "truth_leadlep")
truth_leadw = pd.read_hdf(path, "truth_leadw")
truth_m = pd.read_hdf(path, "truth_m")
truth_met = pd.read_hdf(path, "truth_met")
truth_nu = pd.read_hdf(path, "truth_nu")
truth_subleadlep = pd.read_hdf(path, "truth_subleadlep")
truth_subleadw = pd.read_hdf(path, "truth_subleadw")
truth_w = pd.read_hdf(path, "truth_w")


lead_lep = truth_leadlep[
    [
        "TruthRecoLeadLepPx",
        "TruthRecoLeadLepPy",
        "TruthRecoLeadLepPz",
        "TruthRecoLeadLepE",
    ]
].to_numpy()
lead_w = truth_leadw[
    ["TruthRecoLeadWPx", "TruthRecoLeadWPy", "TruthRecoLeadWPz", "TruthRecoLeadWE"]
].to_numpy()
sublead_lep = truth_subleadlep[
    [
        "TruthRecoSubleadLepPx",
        "TruthRecoSubleadLepPy",
        "TruthRecoSubleadLepPz",
        "TruthRecoSubleadLepE",
    ]
].to_numpy()
sublead_w = truth_subleadw[
    [
        "TruthRecoSubleadWPx",
        "TruthRecoSubleadWPy",
        "TruthRecoSubleadWPz",
        "TruthRecoSubleadWE",
    ]
].to_numpy()


def Bij(particles):
    # Ensure ROOT is properly initialized
    ROOT.gROOT.SetBatch(True)

    def cglmp(z_xp, z_xn, z_yp, z_yn):
        """
        This is a function to calculate Bij (CGLMP values).
        :param z_xp: Angle (xi) between positive lepton and x-axis.
        :param z_xn: Angle (xi) between negative lepton and x-axis.
        :param z_yp: Angle (xi) between positive lepton and y-axis.
        :param z_xn: Angle (xi) between negative lepton and y-axis.
        """
        # count expectation value, use (27) in Alan's paper
        tr_a = (np.divide(8, np.sqrt(3))) * (z_xp * z_xn + z_yp * z_yn)
        tr_b = (
            25
            * (np.square(z_xp) - np.square(z_yp))
            * (np.square(z_xn) - np.square(z_yn))
        )
        tr_c = 100 * (z_xp * z_yp * z_xn * z_yn)
        tr = tr_a + tr_b + tr_c

        return tr

    WpBoson = TLorentzVector(*particles[:4])
    WpLepton = TLorentzVector(*particles[4:8])
    WnBoson = TLorentzVector(*particles[8:12])
    WnLepton = TLorentzVector(*particles[12:16])

    # construct Higgs 4-vector
    Higgs = WpBoson + WnBoson

    # construct a moving orthogonal basis (k,r,n)
    Beam_p = TLorentzVector(0, 0, 1, 1)  # spatial-axis

    # define boost vector
    Higgsb = Higgs.BoostVector()

    # (1) performs a boost transformation from the rod frame to the original one.
    # Perform boost transformation from the rod frame to the original one
    for vec in [WpBoson, WpLepton, WnBoson, WnLepton, Beam_p]:
        vec.Boost(-Higgsb)

    # 2. Define (k,r,n) -> definitions are in Alan's paper
    k_per = TVector3(WpBoson.X(), WpBoson.Y(), WpBoson.Z())
    p_per = TVector3(Beam_p.X(), Beam_p.Y(), Beam_p.Z())  # in the Higgs rest frame
    k = k_per.Unit()  # normalized -> unit vector
    p = p_per.Unit()
    y = p.Dot(k)
    r_length = np.sqrt(1 - y * y)
    r = (1 / r_length) * (p - y * k)
    n = (1 / r_length) * (p.Cross(k))  # (1/sin)*sin = 1 -> unit vector

    # 3. Further boost to W+ and W- frame respectively
    WpkBoost = WpBoson.BoostVector()
    WpBoson.Boost(-WpkBoost)
    WpLepton.Boost(-WpkBoost)
    WnkBoost = WnBoson.BoostVector()
    WnBoson.Boost(-WnkBoost)
    WnLepton.Boost(-WnkBoost)

    # 4. Map all particle to (k,r,n) frame
    WpLp = WpLepton.Vect()  # momentum in (k,r,n)
    WnLp = WnLepton.Vect()
    # Mapping to n-r-k basis
    WpLp_k = TLorentzVector(WpLp.Dot(n), WpLp.Dot(r), WpLp.Dot(k), WpLepton.E())
    WnLp_k = TLorentzVector(WnLp.Dot(n), WnLp.Dot(r), WnLp.Dot(k), WnLepton.E())

    # 5. Calculate directional cosines
    # directional cosine from Wp
    WpLp_Vect_Mag = WpLp_k.Vect().Mag()
    cos_n_join_p = np.divide(WpLp_k.X(), WpLp_Vect_Mag)
    cos_r_join_p = np.divide(WpLp_k.Y(), WpLp_Vect_Mag)
    cos_k_join_p = np.divide(WpLp_k.Z(), WpLp_Vect_Mag)
    # directional cosine from Wn
    WnLp_Vect_Mag = WnLp_k.Vect().Mag()
    cos_n_join_n = np.divide(WnLp_k.X(), WnLp_Vect_Mag)
    cos_r_join_n = np.divide(WnLp_k.Y(), WnLp_Vect_Mag)
    cos_k_join_n = np.divide(WnLp_k.Z(), WnLp_Vect_Mag)

    # 6. Calculate Bij (CGLMP values)
    B_xy = cglmp(cos_n_join_p, cos_n_join_n, cos_r_join_p, cos_r_join_n)
    B_yz = cglmp(cos_r_join_p, cos_r_join_n, cos_k_join_p, cos_k_join_n)
    B_zx = cglmp(cos_n_join_p, cos_n_join_n, cos_k_join_p, cos_k_join_n)

    return [B_xy, B_yz, B_zx]


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
model_lead = tf.keras.models.load_model("./dnn_w_lead_full_scaled.h5")
model_lead.summary()
model_sublead = tf.keras.models.load_model("./dnn_w_sublead_full_scaled.h5")
model_sublead.summary()


lead = pd.read_hdf(path, "reco_leadlep")[
    ["RecoLeadLepE", "RecoLeadLepPx", "RecoLeadLepPy", "RecoLeadLepPz"]
]
sublead = pd.read_hdf(path, "reco_subleadlep")[
    ["RecoSubleadLepE", "RecoSubleadLepPx", "RecoSubleadLepPy", "RecoSubleadLepPz"]
]
met = pd.read_hdf(path, "reco_met")[["RecoMETPx", "RecoMETPy"]]
w_lead = pd.read_hdf(path, "truth_leadw")[
    ["TruthRecoLeadWE", "TruthRecoLeadWPx", "TruthRecoLeadWPy", "TruthRecoLeadWPz"]
]
w_sublead = pd.read_hdf(path, "truth_subleadw")[
    [
        "TruthRecoSubleadWE",
        "TruthRecoSubleadWPx",
        "TruthRecoSubleadWPy",
        "TruthRecoSubleadWPz",
    ]
]

obs_var = pd.concat([lead, sublead, met], axis=1) * GEV
int_var_lead = w_lead * GEV
int_var_sublead = w_sublead * GEV


ROBUST_OBS = RobustScaler()
obs_var = ROBUST_OBS.fit_transform(obs_var)
ROBUST_INT_LEAD = RobustScaler().fit(int_var_lead)
int_var_lead = int_var_lead.to_numpy()
ROBUST_INT_SUBLEAD = RobustScaler().fit(int_var_sublead)
int_var_sublead = int_var_sublead.to_numpy()


pred_int_lead = model_lead.predict(obs_var)
pred_int_lead = ROBUST_INT_LEAD.inverse_transform(pred_int_lead)
pred_int_sublead = model_sublead.predict(obs_var)
pred_int_sublead = ROBUST_INT_SUBLEAD.inverse_transform(pred_int_sublead)


pred_w_lead_energy = pred_int_lead[:, 0]
pred_w_lead_px = pred_int_lead[:, 1]
pred_w_lead_py = pred_int_lead[:, 2]
pred_w_lead_pz = pred_int_lead[:, 3]
pred_w_lead_p4 = np.vstack(
    [pred_w_lead_px, pred_w_lead_py, pred_w_lead_pz, pred_w_lead_energy]
).T
pred_w_sublead_energy = pred_int_sublead[:, 0]
pred_w_sublead_px = pred_int_sublead[:, 1]
pred_w_sublead_py = pred_int_sublead[:, 2]
pred_w_sublead_pz = pred_int_sublead[:, 3]
pred_w_sublead_p4 = np.vstack(
    [pred_w_sublead_px, pred_w_sublead_py, pred_w_sublead_pz, pred_w_sublead_energy]
).T
w_lead_energy = w_lead["TruthRecoLeadWE"] * GEV
w_lead_px = w_lead["TruthRecoLeadWPx"] * GEV
w_lead_py = w_lead["TruthRecoLeadWPy"] * GEV
w_lead_pz = w_lead["TruthRecoLeadWPz"] * GEV
w_sublead_energy = w_sublead["TruthRecoSubleadWE"] * GEV
w_sublead_px = w_sublead["TruthRecoSubleadWPx"] * GEV
w_sublead_py = w_sublead["TruthRecoSubleadWPy"] * GEV
w_sublead_pz = w_sublead["TruthRecoSubleadWPz"] * GEV


particles = np.concatenate(
    [pred_w_lead_p4, lead_lep, pred_w_sublead_p4, sublead_lep], axis=1
)
bij = np.zeros((particles.shape[0], 3))
with multiprocessing.Pool(WORKERS) as pool:
    bij = np.array(pool.map(Bij, particles))
    pool.close()
    pool.join()


mask = np.any(np.isnan(bij), axis=1)
bij_cleaned = bij[~mask, :]


def ci(data):
    res = stats.bootstrap(
        (data,),
        statistic=np.mean,
        confidence_level=0.95,
        n_resamples=1_024,
        vectorized=True,
        alternative="two-sided",
        method="BCa",
        random_state=RNG,
        batch=16_384,
    )
    return res.confidence_interval


with multiprocessing.Pool(WORKERS) as pool:
    reco_ci = pool.map(ci, bij_cleaned.T)
    reco_ci_xy_low, reco_ci_xy_high = reco_ci[0][0], reco_ci[0][1]
    reco_ci_yz_low, reco_ci_yz_high = reco_ci[1][0], reco_ci[1][1]
    reco_ci_zx_low, reco_ci_zx_high = reco_ci[2][0], reco_ci[2][1]
    pool.close()
    pool.join()
print(
    f"""
<RECO>: EVT_NUM: {len(bij_cleaned)}
Bxy: mean = {bij_cleaned[:, 0].mean():<.3f} with 95%CI: -{bij_cleaned[:, 0].mean() - reco_ci_xy_low:<.3f}; +{reco_ci_xy_high - bij_cleaned[:, 0].mean():<.3f}
Byz: mean = {bij_cleaned[:, 1].mean():<.3f} with 95%CI: -{bij_cleaned[:, 1].mean() - reco_ci_yz_low:<.3f}; +{reco_ci_yz_high - bij_cleaned[:, 1].mean():<.3f}
Bzx: mean = {bij_cleaned[:, 2].mean():<.3f} with 95%CI: -{bij_cleaned[:, 2].mean() - reco_ci_zx_low:<.3f}; +{reco_ci_zx_high - bij_cleaned[:, 2].mean():<.3f}"""
)

truth_Bij = truth_Bij.to_numpy()
with multiprocessing.Pool(WORKERS) as pool:
    truth_ci = pool.map(ci, truth_Bij.T)
    truth_ci_xy_low, truth_ci_xy_high = truth_ci[0][0], truth_ci[0][1]
    truth_ci_yz_low, truth_ci_yz_high = truth_ci[1][0], truth_ci[1][1]
    truth_ci_zx_low, truth_ci_zx_high = truth_ci[2][0], truth_ci[2][1]
    pool.close()
    pool.join()
print(
    f"""
<TRUTH>: EVT_NUM: {len(truth_Bij)}
Bxy: mean = {truth_Bij[:, 0].mean():<.3f} with 95%CI: -{truth_Bij[:, 0].mean() - truth_ci_xy_low:<.3f}; +{truth_ci_xy_high - truth_Bij[:, 0].mean():<.3f}
Byz: mean = {truth_Bij[:, 1].mean():<.3f} with 95%CI: -{truth_Bij[:, 1].mean() - truth_ci_yz_low:<.3f}; +{truth_ci_yz_high - truth_Bij[:, 1].mean():<.3f}
Bzx: mean = {truth_Bij[:, 2].mean():<.3f} with 95%CI: -{truth_Bij[:, 2].mean() - truth_ci_zx_low:<.3f}; +{truth_ci_zx_high - truth_Bij[:, 2].mean():<.3f}"""
)


t_end = time.time()
print(f"Total spending time: {t_end-t_start: .2f} (s)")
