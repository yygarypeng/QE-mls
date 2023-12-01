import numpy as np
import pandas as pd
import glob
import uproot

from dataclasses import dataclass


class Read_root:
    def __init__(
        self,
        leaves,
        path="/root/work/truthreco/signal/*mc16d*/",
        tree="CollectionTree;1",
    ):
        self.path = path
        self.tree = tree
        self.leaves = leaves
        self.pattern = glob.glob(self.path + "*.root")[0:1]

    def read_file(self, pat):
        return uproot.open(pat)[self.tree].arrays(self.leaves, library="pandas")

    def gen_data(self):
        container = np.empty(len(self.pattern), dtype=object)
        for ind, pat in enumerate(self.pattern):
            container[ind] = self.read_file(pat)
        return pd.concat(container, axis=0)


@dataclass
class Leaves:
    leaves_truth_lead_lep = []
    leaves_truth_sublead_lep = []
    leaves_truth_met = [
        "MET_TruthAuxDyn.mpx",
        "MET_TruthAuxDyn.mpy",
    ]
    leaves_truth_electron = [
        "HWWTruthElectronsAuxDyn.e",
        "HWWTruthElectronsAuxDyn.px",
        "HWWTruthElectronsAuxDyn.py",
        "HWWTruthElectronsAuxDyn.pz",
    ]
    leaves_truth_muon = [
        "HWWTruthMuonsAuxDyn.e",
        "HWWTruthMuonsAuxDyn.px",
        "HWWTruthMuonsAuxDyn.py",
        "HWWTruthMuonsAuxDyn.pz",
    ]
    leaves_truth_e_neu = []
    leaves_truth_mu_neu = []

    leaves_reco_lead_lep = []
    leaves_reco_sublead_lep = []
    leaves_track_met = [
        "HWWTrackMETAuxDyn.mpx",
        "HWWTrackMETAuxDyn.mpy",
    ]
    leaves_reco_met = [
        "HWWMETAuxDyn.mpx",
        "HWWMETAuxDyn.mpy",
    ]
    leaves_reco_electron = [
        # "HWWElectronsAuxDyn.m",
        "HWWElectronsAuxDyn.pt",
        "HWWElectronsAuxDyn.eta",
        "HWWElectronsAuxDyn.phi",
    ]
    leaves_reco_muon = [
        # "HWWMuonsAuxDyn.m",
        "HWWMuonsAuxDyn.pt",
        "HWWMuonsAuxDyn.eta",
        "HWWMuonsAuxDyn.phi",
    ]
    leaves_reco_e_neu = []
    leaves_reco_mu_neu = []


@dataclass
class Data:
    leaves = Leaves()

    _reader_e_truth = Read_root(leaves.leaves_truth_electron)
    e_truth: pd.DataFrame = _reader_e_truth.gen_data()

    _reader_mu_truth = Read_root(leaves.leaves_truth_muon)
    mu_truth: pd.DataFrame = _reader_mu_truth.gen_data()

    _reader_met_truth = Read_root(leaves.leaves_truth_met)
    met_truth: pd.DataFrame = _reader_met_truth.gen_data()

    _reader_e_reco = Read_root(leaves.leaves_reco_electron)
    e_reco: pd.DataFrame = _reader_e_reco.gen_data()

    _reader_mu_reco = Read_root(leaves.leaves_reco_muon)
    mu_reco: pd.DataFrame = _reader_mu_reco.gen_data()

    _reader_met_reco = Read_root(leaves.leaves_reco_met)
    met_reco: pd.DataFrame = _reader_met_reco.gen_data()


import unittest
from get_root_data import Leaves


class TestLeaves(unittest.TestCase):
    def setUp(self):
        self.leaves = Leaves()

    def test_truth_met(self):
        self.assertEqual(
            self.leaves.leaves_truth_met, ["MET_TruthAuxDyn.mpx", "MET_TruthAuxDyn.mpy"]
        )

    def test_truth_electron(self):
        self.assertEqual(
            self.leaves.leaves_truth_electron,
            [
                "HWWTruthElectronsAuxDyn.e",
                "HWWTruthElectronsAuxDyn.px",
                "HWWTruthElectronsAuxDyn.py",
                "HWWTruthElectronsAuxDyn.pz",
            ],
        )

    def test_truth_muon(self):
        self.assertEqual(
            self.leaves.leaves_truth_muon,
            [
                "HWWTruthMuonsAuxDyn.e",
                "HWWTruthMuonsAuxDyn.px",
                "HWWTruthMuonsAuxDyn.py",
                "HWWTruthMuonsAuxDyn.pz",
            ],
        )

    def test_reco_met(self):
        self.assertEqual(
            self.leaves.leaves_reco_met, ["HWWMETAuxDyn.mpx", "HWWMETAuxDyn.mpy"]
        )

    def test_reco_electron(self):
        self.assertEqual(
            self.leaves.leaves_reco_electron,
            [
                "HWWElectronsAuxDyn.pt",
                "HWWElectronsAuxDyn.eta",
                "HWWElectronsAuxDyn.phi",
            ],
        )

    def test_reco_muon(self):
        self.assertEqual(
            self.leaves.leaves_reco_muon,
            ["HWWMuonsAuxDyn.pt", "HWWMuonsAuxDyn.eta", "HWWMuonsAuxDyn.phi"],
        )


if __name__ == "__main__":
    unittest.main()
