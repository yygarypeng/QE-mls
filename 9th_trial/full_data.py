from dataclasses import dataclass
import pandas as pd
import numpy as np

data_path = "/root/data/recotruth/Truth_Reco_345324_multi_rtag_w.h5"
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
    px = pd.read_hdf(data_path, "RecoLep0")["Px"] * GEV
    py = pd.read_hdf(data_path, "RecoLep0")["Py"] * GEV
    pz = pd.read_hdf(data_path, "RecoLep0")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "RecoLep0")["E"] * GEV
    pt = pd.read_hdf(data_path, "RecoLep0")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "RecoLep0")["Eta"]
    phi = pd.read_hdf(data_path, "RecoLep0")["Phi"]
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Sublead_lep:
    px = pd.read_hdf(data_path, "RecoLep1")["Px"] * GEV
    py = pd.read_hdf(data_path, "RecoLep1")["Py"] * GEV
    pz = pd.read_hdf(data_path, "RecoLep1")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "RecoLep1")["E"] * GEV
    pt = pd.read_hdf(data_path, "RecoLep1")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "RecoLep1")["Eta"]
    phi = pd.read_hdf(data_path, "RecoLep1")["Phi"]
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
    px = pd.read_hdf(data_path, "RecoMET")["Px"] * GEV
    py = pd.read_hdf(data_path, "RecoMET")["Py"] * GEV
    phi = pd.read_hdf(data_path, "RecoMET")["Phi"]
    pt = pt(px, py)


@dataclass
class Truth_lead_lep:
    px = pd.read_hdf(data_path, "TruthLep0")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthLep0")["Py"] * GEV
    pz = pd.read_hdf(data_path, "TruthLep0")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "TruthLep0")["E"] * GEV
    pt = pd.read_hdf(data_path, "TruthLep0")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "TruthLep0")["Eta"]
    phi = pd.read_hdf(data_path, "TruthLep0")["Phi"]
    p4 = np.array([px, py, pz, energy]).T
    p3 = np.array([px, py, pz]).T


@dataclass
class Truth_sublead_lep:
    px = pd.read_hdf(data_path, "TruthLep1")["Px"] * GEV
    py = pd.read_hdf(data_path, "TruthLep1")["Py"] * GEV
    pz = pd.read_hdf(data_path, "TruthLep1")["Pz"] * GEV
    energy = pd.read_hdf(data_path, "TruthLep1")["E"] * GEV
    pt = pd.read_hdf(data_path, "TruthLep1")["Pt"] * GEV
    eta = pd.read_hdf(data_path, "TruthLep1")["Eta"]
    phi = pd.read_hdf(data_path, "TruthLep1")["Phi"]
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


@dataclass
class MC_weight:
    w = pd.read_hdf(data_path, "mcWeight")


if __name__ == "__main__":
    print(Lead_lep)
    print(Sublead_lep)
    print(Dilep)
    print(Met)
    print(Truth_lead_lep)
    print(Truth_sublead_lep)
    print(Truth_dilep)
    print(Truth_met)
    print(Lead_w)
    print(Sublead_w)
    print(MC_weight)
