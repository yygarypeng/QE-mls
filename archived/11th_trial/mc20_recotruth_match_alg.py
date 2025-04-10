import glob
import uproot
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

class RecoTruthMatch_alg:
    def __init__(self, nTuple_file):
        self.nTuple_file = nTuple_file
    
    def process_ntuple_files(self, branches, data_mode="truth"):
        '''
        :params branches: list of branches to extract
        :params data_mode: 'truth' or 'reco' to extract truth or reco data
        
        :return: dictionary of data from nTuple files
        '''
        # Initialize dictionary to accumulate results for each branch
        accumulated_results = {branch: [] for branch in branches}

        # Load and extract data from each file
        with uproot.open(self.nTuple_file, mode="r") as file:
            data = file[data_mode + ";1"].arrays(branches, library="np")
            evt_num = data[branches[0]].shape[0]

        for branch in branches:
            accumulated_results[branch].append(data[branch])

        # Process each branch (without multiprocessing)
        results = []
        for branch in tqdm(branches):
            # Concatenate arrays for this branch
            concatenated = np.concatenate(accumulated_results[branch])
            results.append((branch, concatenated))

        # Convert results back to dictionary
        return dict(results)

    def extract_variables(self, data, branches):
        '''
        :params data: dictionary of data from process_ntuple_files
        :params branches: list of branches to extract
        
        :return: list of extracted variables
        '''
        return [data[branch] for branch in branches]

    def is_valid(self, part1, part2, part11, part12, part21, part22):
        '''
        :params part1 : pdgId of the Hdecay1 --> W+
        :params part2 : pdgId of the Hdecay2 --> W-
        :params part11: pdgId of the particle 1 from Hdecay1 --> neutrino
        :params part12: pdgId of the particle 2 from Hdecay1 --> lepton
        :params part21: pdgId of the particle 1 from Hdecay2 --> lepton
        :params part22: pdgId of the particle 2 from Hdecay2 --> neutrino
        
        :return: boolean array of valid H->WW*->lvlv events
        '''
        cond1 = (part11 * part12 == -11*12) | (part11 * part12 == -13*14)  # pairing correct l and nu for part1
        cond2 = (part21 * part22 == -11*12) | (part21 * part22 == -13*14)  # pairing correct l and nu for part2
        cond3 = part12 * part21 == -11*13  # pairing diff flav leps and conserving charge (e, mu only)
        cond4 = part1 == 24   # confirm W+ boson (excluding zero paddings)
        cond5 = part2 == -24  # confirm W- boson (excluding zero paddings)
        # higgs has already been checked to be the correct particle ID form previous plots
        return cond1 & cond2 & cond3 & cond4 & cond5 

    def match_events(self, truth_eventNumber, reco_eventNumber, valid_ind):
        '''
        :params truth_eventNumber: eventNumber from truth data
        :params reco_eventNumber: eventNumber from reco data
        :params valid_ind: boolean array of valid H->WW*->lvlv events (pass is_valid func)
        '''
        matched_ind = np.intersect1d(truth_eventNumber[valid_ind], reco_eventNumber)
        truth_matched_ind = np.isin(truth_eventNumber[valid_ind], matched_ind)
        reco_matched_ind = np.isin(reco_eventNumber, matched_ind)
        
        if np.sum(truth_matched_ind) != np.sum(reco_matched_ind):
            print("Match truth:", np.sum(truth_matched_ind))
            print("Match reco:", np.sum(reco_matched_ind))
            print("Matched ratio:", np.sum(reco_matched_ind) / np.sum(truth_matched_ind))
            print('Matched reco ratio:', len(matched_ind) / np.sum(reco_matched_ind))
            print('Matched truth ratio:', len(matched_ind) / np.sum(truth_matched_ind))
            print('Matched events (precheck):', len(matched_ind))
            raise ValueError("Truth and reco matched indices do not match")
        else:
            print("Matched ratio:", np.sum(reco_matched_ind) / np.sum(truth_matched_ind))
            print('Matched reco ratio:', len(matched_ind) / np.sum(reco_matched_ind))
            print('Matched truth ratio:', len(matched_ind) / np.sum(truth_matched_ind))
            print('Matched events (precheck):', len(matched_ind))
            print('Truth and reco matched indices match!')

        if (np.where(np.unique(truth_eventNumber[valid_ind][truth_matched_ind], return_counts=True)[1] == 2)[0].shape[0] != 0) |\
            (np.where(np.unique(reco_eventNumber[reco_matched_ind], return_counts=True)[1] == 2)[0].shape[0] != 0):
            print(np.unique(np.unique(truth_eventNumber[valid_ind][truth_matched_ind], return_counts=True)[1]))
            truth_evtnum_2num = np.where(np.unique(truth_eventNumber[valid_ind][truth_matched_ind], return_counts=True)[1] == 2)
            print(np.unique(np.unique(reco_eventNumber[reco_matched_ind], return_counts=True)[1]))
            reco_evtnum_2num = np.where(np.unique(reco_eventNumber[reco_matched_ind], return_counts=True)[1] == 2)
            print(len(reco_eventNumber))
            raise ValueError("There are events with multiple matches")
        else:
            print('No multiple matches found!')
        
        return matched_ind, truth_matched_ind, reco_matched_ind
    
    def get_matched_events(self, truth_eventNumber, reco_eventNumber, valid_ind):
        matched_ind, truth_matched_ind, reco_matched_ind = self.match_events(truth_eventNumber, reco_eventNumber, valid_ind)
        matched_truth_events = truth_eventNumber[valid_ind][truth_matched_ind]
        matched_reco_events = reco_eventNumber[reco_matched_ind]
        return matched_truth_events, matched_reco_events


# Example usage
if __name__ == "__main__":
    # Define branches
    truth_branches = [
        'HWW_MC_Hdecay1_decay1_afterFSR_pt',
        'HWW_MC_Hdecay1_decay1_afterFSR_eta',
        'HWW_MC_Hdecay1_decay1_afterFSR_phi',
        'HWW_MC_Hdecay1_decay1_afterFSR_pdgId',
        'HWW_MC_Hdecay1_decay2_afterFSR_pt',
        'HWW_MC_Hdecay1_decay2_afterFSR_eta',
        'HWW_MC_Hdecay1_decay2_afterFSR_phi',
        'HWW_MC_Hdecay1_decay2_afterFSR_pdgId',
        'HWW_MC_Hdecay2_decay1_afterFSR_pt',
        'HWW_MC_Hdecay2_decay1_afterFSR_eta',
        'HWW_MC_Hdecay2_decay1_afterFSR_phi',
        'HWW_MC_Hdecay2_decay1_afterFSR_pdgId',
        'HWW_MC_Hdecay2_decay2_afterFSR_pt',
        'HWW_MC_Hdecay2_decay2_afterFSR_eta',
        'HWW_MC_Hdecay2_decay2_afterFSR_phi',
        'HWW_MC_Hdecay2_decay2_afterFSR_pdgId',
        'HWW_MC_Hdecay1_afterFSR_pt',
        'HWW_MC_Hdecay1_afterFSR_eta',
        'HWW_MC_Hdecay1_afterFSR_phi',
        'HWW_MC_Hdecay1_afterFSR_m',
        'HWW_MC_Hdecay1_afterFSR_pdgId',
        'HWW_MC_Hdecay2_afterFSR_pt',
        'HWW_MC_Hdecay2_afterFSR_eta',
        'HWW_MC_Hdecay2_afterFSR_phi',
        'HWW_MC_Hdecay2_afterFSR_m',
        'HWW_MC_Hdecay2_afterFSR_pdgId',
        'HWW_MC_H_afterFSR_pt',
        'HWW_MC_H_afterFSR_eta',
        'HWW_MC_H_afterFSR_phi',
        'HWW_MC_H_afterFSR_m',
        'HWW_MC_H_afterFSR_pdgId',
        'eventNumber',
    ]

    # Define nTuple_dir_list
    nTuple_dir_list = glob.glob("/root/data/qe-stkorn-v2/*/*.root")[0]

    # Create an instance of RecoTruthMatch_alg
    processor = RecoTruthMatch_alg(nTuple_dir_list)

    # Process files and extract variables
    truth_mixing_data = processor.process_ntuple_files(truth_branches, data_mode="truth")
    truth_mixing_data = processor.extract_variables(truth_mixing_data, truth_branches)
    
    (
        truth_lnu11_pt, truth_lnu11_eta, truth_lnu11_phi, truth_lnu11_id,
        truth_lnu12_pt, truth_lnu12_eta, truth_lnu12_phi, truth_lnu12_id,
        truth_lnu21_pt, truth_lnu21_eta, truth_lnu21_phi, truth_lnu21_id,
        truth_lnu22_pt, truth_lnu22_eta, truth_lnu22_phi, truth_lnu22_id,
        truth_w1_pt, truth_w1_eta, truth_w1_phi, truth_w1_m, truth_w1_id,
        truth_w2_pt, truth_w2_eta, truth_w2_phi, truth_w2_m, truth_w2_id,
        truth_higgs_pt, truth_higgs_eta, truth_higgs_phi, truth_higgs_m, truth_higgs_id,
        truth_eventNumber,
    ) = truth_mixing_data

    print('Truth sample length:', len(truth_higgs_pt))

    # Define valid indices (example, you need to define valid_ind based on your logic)
    valid_ind = processor.is_valid(truth_w1_id, truth_w2_id, truth_lnu11_id, truth_lnu12_id, truth_lnu21_id, truth_lnu22_id)

    # Define reco branches and process reco data
    reco_branches = [
        'met_met_NOSYS',
        'met_phi_NOSYS',
        'el_pt_NOSYS',
        'el_eta',
        'el_phi',
        'mu_pt_NOSYS',
        'mu_eta',
        'mu_phi',
        'll_m_NOSYS',
        'll_deta_NOSYS',
        'll_dphi_NOSYS',
        'eventNumber'
    ]

    reco_data = processor.process_ntuple_files(reco_branches, data_mode='reco')
    reco_data = processor.extract_variables(reco_data, reco_branches)

    # Perform matching and get matched events
    (
        met_pt, met_phi, 
        el_pt, el_eta, el_phi,
        mu_pt, mu_eta, mu_phi,
        ll_m, ll_deta, ll_dphi,
        reco_eventNumber,
    ) = reco_data
    matched_truth_events, matched_reco_events = processor.get_matched_events(truth_eventNumber, reco_eventNumber, valid_ind)
    print("Matching successful!")

    print("Matched truth events:", matched_truth_events)
    print("Matched reco events:", matched_reco_events)
    

    print('Reco sample length:', len(met_pt))
    truth_mixing_data = truth_mixing_data[truth_eventNumber == matched_truth_events]
    reco_data = reco_data[reco_eventNumber == matched_reco_events]
    print('Truth sample :', truth_mixing_data[0])
    print('Reco sample :', reco_data[0])
    