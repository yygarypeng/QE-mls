import gc
import glob
import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self):
        self.GEV = 1e3
        self.RMV_EVT = []

    def load_files(self, path):
        files_name = self.get_files_names(path)
        self.files_name = files_name
        self.files_name.sort()
        self.files = [self.get_data(f) for f in self.files_name]
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
            return pd.DataFrame()

    def process_lep(self, LepP, LepM):
        lep_kin = (
            pd.DataFrame(
                {
                    "lep_p_E": LepP["E"],
                    "lep_p_px": LepP["px"],
                    "lep_p_py": LepP["py"],
                    "lep_p_pz": LepP["pz"],
                    "lep_m_E": LepM["E"],
                    "lep_m_px": LepM["px"],
                    "lep_m_py": LepM["py"],
                    "lep_m_pz": LepM["pz"],
                }
            )
            / self.GEV
        )
        lep_kin.drop(self.RMV_EVT, inplace=True)
        return lep_kin

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

    def process_dinu(self, NuP, NuM):
        # Kinemetic info of neutirnos.
        nu_kin = (
            pd.DataFrame(
                {
                    "nu_E": NuP["E"] + NuM["E"],
                    "nu_px": NuP["px"] + NuM["px"],
                    "nu_py": NuP["py"] + NuM["py"],
                    "nu_pz": NuP["pz"] + NuM["pz"],
                }
            )
            / self.GEV
        )
        nu_kin.drop(self.RMV_EVT, inplace=True)
        return nu_kin

    def process_nu(self, Nu):
        # Kinemetic info of neutirnos.
        nu_kin = (
            pd.DataFrame(
                {
                    "nu_E": Nu["E"],
                    "nu_px": Nu["px"],
                    "nu_py": Nu["py"],
                    "nu_pz": Nu["pz"],
                }
            )
            / self.GEV
        )
        nu_kin.drop(self.RMV_EVT, inplace=True)
        return nu_kin

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

    (
        CGLMP,
        Higgs,
        LeadLep,
        LepM,
        LepP,
        NuM,
        NuP,
        MET,
        Wm,
        Wp,
        diLep,
        SubLep,
        Xi,
    ) = processor.files

    lep_kin = processor.process_lep(LepP, LepM)
    print("lep_kin shape:", lep_kin.shape)
    lep_kin.head(5)

    MET_kin = processor.process_MET(MET)
    print("MET_kin shape:", MET_kin.shape)
    MET_kin.head(5)

    dinu_kin = processor.process_dinu(NuP, NuM)
    print("dinu_kin shape:", dinu_kin.shape)
    dinu_kin.head(5)

    CGLMP_kin = processor.process_CGLMP(CGLMP)
    print("CGLMP shape:", CGLMP_kin.shape)
    CGLMP_kin.head(5)

    del processor  # Clear the instance
    gc.collect()
