from dataclasses import dataclass
import pandas as pd
import numpy as np

data_path = "/root/data/recotruth/Truth_Reco_345324_multi_rtag.h5"
GEV = 1e-3

def pt(px, py):
    return np.sqrt(np.square(px) + np.square(py))


def eta(px, py, pz):
    pt = np.sqrt(np.square(px) + np.square(py))
    return np.arcsinh(np.divide(pz, pt))


def phi(px, py):
    return np.arctan2(py, px)


def m(p4):
    return np.sqrt(
        np.square(p4[:, 3])
        - np.square(p4[:, 0])
        - np.square(p4[:, 1])
        - np.square(p4[:, 2])
    )


@dataclass
class Lead_lep:
    px = pd.read_hdf(data_path, "RecoCandLep0")["Px"] * GEV
    py = pd.read_hdf(data_path, "RecoCandLep0")["Py"] * GEV
    pz = pd.read_hdf(data_path, "RecoCandLep0")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "RecoCandLep0")["E"] * GEV
    pt = pd.read_hdf(data_path, "RecoCandLep0")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "RecoCandLep0")["Eta"]
    phi = pd.read_hdf(data_path, "RecoCandLep0")["Phi"]
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Sublead_lep:
    px = pd.read_hdf(data_path, "RecoCandLep1")["Px"] * GEV
    py = pd.read_hdf(data_path, "RecoCandLep1")["Py"] * GEV
    pz = pd.read_hdf(data_path, "RecoCandLep1")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "RecoCandLep1")["E"] * GEV
    pt = pd.read_hdf(data_path, "RecoCandLep1")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "RecoCandLep1")["Eta"]
    phi = pd.read_hdf(data_path, "RecoCandLep1")["Phi"]
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Dilep:
    lead = Lead_lep()
    sublead = Sublead_lep()
    p4 = Lead_lep.p4 + Sublead_lep.p4
    p3 = p4[:, :3]
    px = p4[:, 0]
    py = p4[:, 1]
    pz = p4[:, 2]
    energy = p4[:, 3]
    pt = pt(px, py)
    eta = eta(px, py, pz)
    phi = phi(px, py)
    m = m(p4)


@dataclass
class Met:
    px = pd.read_hdf(data_path, "RecoCandMET")["Px"] * GEV
    py = pd.read_hdf(data_path, "RecoCandMET")["Py"] * GEV
    phi = pd.read_hdf(data_path, "RecoCandMET")["Phi"]
    pt = pt(px, py)


@dataclass
class Truth_lead_lep:
    px = pd.read_hdf(data_path, "TruthCandLepn")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthCandLepn")["Py"] * GEV
    pz = pd.read_hdf(data_path, "TruthCandLepn")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "TruthCandLepn")["E"] * GEV
    pt = pd.read_hdf(data_path, "TruthCandLepn")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "TruthCandLepn")["Eta"]
    phi = pd.read_hdf(data_path, "TruthCandLepn")["Phi"]
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Truth_sublead_lep:
    px = pd.read_hdf(data_path, "TruthCandLepp")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthCandLepp")["Py"] * GEV
    pz = pd.read_hdf(data_path, "TruthCandLepp")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "TruthCandLepp")["E"] * GEV
    pt = pd.read_hdf(data_path, "TruthCandLepp")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "TruthCandLepp")["Eta"]
    phi = pd.read_hdf(data_path, "TruthCandLepp")["Phi"]
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Truth_dilep:
    lead = Truth_lead_lep()
    sublead = Truth_sublead_lep()
    p4 = Lead_lep.p4 + Sublead_lep.p4
    p3 = p4[:, :3]
    px = p4[:, 0]
    py = p4[:, 1]
    pz = p4[:, 2]
    energy = p4[:, 3]
    pt = pt(px, py)
    eta = eta(px, py, pz)
    phi = phi(px, py)
    m = m(p4)


@dataclass
class Truth_met:
    px = pd.read_hdf(data_path, "TruthCandMET")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthCandMET")["Py"] * GEV
    phi = pd.read_hdf(data_path, "TruthCandMET")["Phi"]
    pt = pt(px, py)


@dataclass
class Lead_w:
    px = pd.read_hdf(data_path, "TruthW0")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthW0")["Py"] * GEV
    pz = pd.read_hdf(data_path, "TruthW0")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "TruthW0")["E"] * GEV
    pt = pd.read_hdf(data_path, "TruthW0")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "TruthW0")["Eta"]
    phi = pd.read_hdf(data_path, "TruthW0")["Phi"]
    m = pd.read_hdf(data_path, "TruthW0")["M"] * GEV
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Sublead_w:
    px = pd.read_hdf(data_path, "TruthW1")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthW1")["Py"] * GEV
    pz = pd.read_hdf(data_path, "TruthW1")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "TruthW1")["E"] * GEV
    pt = pd.read_hdf(data_path, "TruthW1")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "TruthW1")["Eta"]
    phi = pd.read_hdf(data_path, "TruthW1")["Phi"]
    m = pd.read_hdf(data_path, "TruthW1")["M"] * GEV
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T