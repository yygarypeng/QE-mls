import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
import multiprocessing
import os

import unittest
import tempfile


@dataclass
class Data:
    CGLMP: pd.DataFrame
    Higgs: pd.DataFrame
    LeadLep: pd.DataFrame
    LepM: pd.DataFrame
    LepP: pd.DataFrame
    NuM: pd.DataFrame
    NuP: pd.DataFrame
    MET: pd.DataFrame
    Wm: pd.DataFrame
    Wp: pd.DataFrame
    diLep: pd.DataFrame
    SubLep: pd.DataFrame
    Xi: pd.DataFrame


class DataProcessor:
    def __init__(self, sampling=int(1e3), processor=os.cpu_count(), random_seed=42):
        self.used_processes: int = processor
        self.sampling: int = sampling
        self.rng = np.random.default_rng(random_seed)
        self.GEV = 1e3
        self.RMV_EVT = []

    def load_files(self, path):
        files_name = self.get_files_names(path)
        self.files_name = files_name
        self.files_name.sort()

        num_processes = multiprocessing.cpu_count()
        print(f"Number of available processors: {num_processes}")
        print(f"Number of used processors: {self.used_processes}")
        print()
        with multiprocessing.Pool(self.used_processes) as p:
            self.files = p.map(self.get_data, self.files_name)

        print(files_name)
        print()

    def get_files_names(self, path):
        files_name = glob.glob(path)
        return files_name

    def get_data(self, path):
        with np.load(path, allow_pickle=True) as f:
            data_dict = {name: f[name] for name in f.files}
        return pd.DataFrame(data_dict)

    def process_part(self, part):
        part_kin = (
            pd.DataFrame(
                {
                    "E": part["E"],
                    "px": part["px"],
                    "py": part["py"],
                    "pz": part["pz"],
                    "m": part["m"],
                    "pt": part["pt"],
                    "eta": part["eta"],
                    "phi": part["phi"],
                }
            )
            / self.GEV
        )
        part_kin.drop(self.RMV_EVT, inplace=True)
        return part_kin

    def process_MET(self, MET):
        nu_kin = (
            pd.DataFrame(
                {
                    "MET_E": np.sqrt(
                        np.square(34141 * np.ones(len(MET)))
                        + np.square(MET["px"])
                        + np.square(MET["py"])
                    ),
                    "MET_px": MET["px"],
                    "MET_py": MET["py"],
                    "MET_pz": np.zeros(len(MET)),
                }
            )
            / self.GEV
        )
        nu_kin.drop(self.RMV_EVT, inplace=True)
        return nu_kin

    def process_dipart(self, part1, part2):
        # Kinemetic info of neutirnos.
        dipart_kin = (
            pd.DataFrame(
                {
                    "E": part1["E"] + part2["E"],
                    "px": part1["px"] + part2["px"],
                    "py": part1["py"] + part2["py"],
                    "pz": part1["pz"] + part2["pz"],
                    "m": np.sqrt(
                        np.square(part1["E"] + part2["E"])
                        - np.square(part1["px"] + part2["px"])
                        - np.square(part1["py"] + part2["py"])
                        - np.square(part1["pz"] + part2["pz"])
                    ),
                }
            )
            / self.GEV
        )
        dipart_kin.drop(self.RMV_EVT, inplace=True)
        return dipart_kin

    def process_CGLMP(self, CGLMP):
        CGLMP = pd.DataFrame(
            {
                "Bxy": CGLMP["Bxy"],
                "Byz": CGLMP["Byz"],
                "Bzx": CGLMP["Bzx"],
            }
        )
        CGLMP.drop(self.RMV_EVT, inplace=True)
        return CGLMP


#  Test codes
class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.dp = DataProcessor(sampling=1000)
        self.dp.RMV_EVT = []  # Ensure RMV_EVT is empty

    def test_get_files_names(self):
        # Replace with a valid path in your system
        path = "/path/to/your/files/*"
        files = self.dp.get_files_names(path)
        self.assertIsInstance(files, list)

    def test_get_data(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp:
            # Save some data to the temporary file
            np.savez(temp.name, np.array([1, 2, 3]))
            # Use the temporary file in the test
            data = self.dp.get_data(temp.name)
            self.assertIsInstance(data, pd.DataFrame)

    def test_process_part(self):
        part = pd.DataFrame(
            {
                "E": np.random.rand(10),
                "px": np.random.rand(10),
                "py": np.random.rand(10),
                "pz": np.random.rand(10),
                "m": np.random.rand(10),
                "pt": np.random.rand(10),
                "eta": np.random.rand(10),
                "phi": np.random.rand(10),
            }
        )
        processed_part = self.dp.process_part(part)
        self.assertIsInstance(processed_part, pd.DataFrame)

    def test_process_MET(self):
        MET = pd.DataFrame(
            {
                "px": np.random.rand(10),
                "py": np.random.rand(10),
            }
        )
        processed_MET = self.dp.process_MET(MET)
        self.assertIsInstance(processed_MET, pd.DataFrame)

    def test_process_dipart(self):
        part1 = pd.DataFrame(
            {
                "E": np.random.rand(10),
                "px": np.random.rand(10),
                "py": np.random.rand(10),
                "pz": np.random.rand(10),
                "m": np.random.rand(10),
                "pt": np.random.rand(10),
                "eta": np.random.rand(10),
                "phi": np.random.rand(10),
            }
        )
        part2 = pd.DataFrame(
            {
                "E": np.random.rand(10),
                "px": np.random.rand(10),
                "py": np.random.rand(10),
                "pz": np.random.rand(10),
                "m": np.random.rand(10),
                "pt": np.random.rand(10),
                "eta": np.random.rand(10),
                "phi": np.random.rand(10),
            }
        )
        processed_dipart = self.dp.process_dipart(part1, part2)
        self.assertIsInstance(processed_dipart, pd.DataFrame)

    def test_process_CGLMP(self):
        CGLMP = pd.DataFrame(
            {
                "Bxy": np.random.rand(10),
                "Byz": np.random.rand(10),
                "Bzx": np.random.rand(10),
            }
        )
        processed_CGLMP = self.dp.process_CGLMP(CGLMP)
        self.assertIsInstance(processed_CGLMP, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
