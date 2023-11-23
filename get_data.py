import gc
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from multiprocessing import Pool


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
    def __init__(self):
        self.GEV = 1e3
        self.RMV_EVT = []

    def load_files(self, path):
        files_name = self.get_files_names(path)
        self.files_name = files_name
        self.files_name.sort()

        with Pool() as p:
            self.files = p.map(self.get_data, self.files_name)

        print(files_name)
        print()

    def get_files_names(self, path):
        files_name = glob.glob(path)
        return files_name

    def get_data(self, path):
        try:
            with np.load(path, allow_pickle=True) as f:
                data_dict = {name: f[name] for name in f.files}
                return pd.DataFrame(data_dict)
        except FileNotFoundError:
            print("File not found!")

    def process_part(self, part):
        part_kin = (
            pd.DataFrame(
                {
                    "E": part["E"],
                    "px": part["px"],
                    "py": part["py"],
                    "pz": part["pz"],
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


if __name__ == "__main__":
    processor = DataProcessor()
    path = "/root/work/truth/signal/*npz"
    processor.load_files(path)
    # Create an instance of the Data dataclass
    data = Data(*processor.files)

    # Now you can access the dataframes like this:
    lep_p = processor.process_part(data.LepP)
    lep_m = processor.process_part(data.LepM)
    lep_kin = pd.concat([lep_p, lep_m], axis=1)
    print("lep_kin shape:", lep_kin.shape)
    lep_kin.head(5)

    MET_kin = processor.process_MET(data.MET)
    print("MET_kin shape:", MET_kin.shape)
    MET_kin.head(5)

    dinu_kin = processor.process_dipart(data.NuP, data.NuM)
    print("dinu_kin shape:", dinu_kin.shape)
    dinu_kin.head(5)

    CGLMP_kin = processor.process_CGLMP(data.CGLMP)
    print("CGLMP shape:", CGLMP_kin.shape)
    CGLMP_kin.head(5)

    del processor  # Clear the instance
    gc.collect()
