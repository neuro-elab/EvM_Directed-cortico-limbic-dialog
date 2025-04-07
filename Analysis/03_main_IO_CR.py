import os
import numpy as np
import time
import pandas as pd
import sys
import glob
import warnings
import h5py
analysis_path = os.path.dirname(os.path.dirname(__file__))

sys.path.insert(0, analysis_path)
gf_dir = os.path.join(analysis_path, 'general_funcs')
sys.path.insert(0, gf_dir)
st_dir = os.path.join(analysis_path, 'sig_trial')
sys.path.insert(0, gf_dir)
gf_dir = os.path.join(analysis_path, 'IO')
sys.path.insert(0, gf_dir)
gf_dir = os.path.join(analysis_path, 'IO', 'IO_funcs')
sys.path.insert(0, gf_dir)
import re
from ExI_funcs import ExI_func_across as IOf_across
from general_funcs import basic_func as bf
from general_funcs import load_summary as ls
import statsmodels

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']

folder = 'InputOutput'
cond_folder = 'CR'
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
tab_region = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='plot_Community')
tab_region = tab_region.sort_values('Order').reset_index(drop=True)
regions_all = tab_region.Area.values
region_col = tab_region.color.values

CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')

class main:
    def __init__(self, subj):
        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.folder = 'InputOutput'
        self.cond_folder = 'CR'
        self.path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        path_gen = os.path.join(sub_path + '\\Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
        path_infos = os.path.join(path_gen, 'Electrodes')
        if not os.path.exists(os.path.join(path_infos, subj + "_labels.xlsx")):
            path_infos = os.path.join(path_gen, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'

        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        if "type" in lbls:
            lbls = lbls[lbls.type == 'SEEG']
            lbls = lbls.reset_index(drop=True)
        self.labels_all = lbls.label.values
        self.labels_C = lbls.Clinic.values
        self.hemisphere = lbls.Hemisphere
        stimlist = pd.read_csv(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        if len(stimlist) == 0:
            stimlist = pd.read_csv(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        #
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        bad_region = np.where((labels_region == 'Unknown') | (labels_region == 'WM') | (labels_region == 'OUT') | (
                labels_region == 'Putamen'))[0]
        self.labels_region_L = lbls.Hemisphere.values + '_' + labels_region
        self.subj = subj
        self.lbls = lbls
        self.coord_all= coord_all

        self.labels_region = labels_region

        self.regions = regions_all
        badchans = pd.read_csv(self.path_patient_analysis + '/BrainMapping/data/badchan.csv')
        self.bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
        # C = regions.label.values
        # self.path_patient   = path_patient
        # self.path_patient_analysis = os.path.join(os.path.dirname(os.path.dirname(self.path_patient)), 'Projects\EL_experiment\Analysis\Patients', subj)
        ##bad channels
        WM_chans = np.where(self.labels_region == 'WM')[0]
        self.bad_all = np.unique(np.concatenate([WM_chans, bad_region, self.bad_chans])).astype('int')
        stim_chans = np.arange(len(labels_all))
        self.stim_chans = np.delete(stim_chans, self.bad_all, 0)

    def get_AUC_mean(self, con_trial, EEG_resp, skip):

        con_trial = con_trial[~np.isin(con_trial.Chan, self.bad_all)].reset_index(drop=True)
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean.csv'  # summary_general
        if os.path.isfile(summary_gen_path) * skip:
            print('AUC-mean already calculated  -  skipping .. ')
        else:
            df = IOf_across.save_AUC_connection(con_trial, EEG_resp)
            df = ls.adding_distance(df, self.coord_all)
            df = ls.adding_area(df, self.lbls, pair=1)
            df = ls.adding_region(df, pair=1)
            df = ls.adding_subregion(df, pair=1)
            df.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake

    def add_delay(self, con_trial, EEG_resp):
        con_trial = con_trial[
            ~np.isin(con_trial.Chan, self.bad_all)].reset_index(
            drop=True)

        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean.csv'  # summary_general
        auc_summary = pd.read_csv(summary_gen_path)
        auc_summary = IOf_across.get_delay(con_trial, EEG_resp, auc_summary)
        auc_summary.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake


    def get_AUC_sleepstate_mean(self, con_trial, EEG_resp, skip = 1):
        SleepStates_val = ['Wake', 'NREM', 'REM']
        con_trial = con_trial[
            ~np.isin(con_trial.Chan, self.bad_all) & np.isin(con_trial.SleepState, SleepStates_val)].reset_index(
            drop=True)

        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean.csv'  # summary_general
        auc_summary = pd.read_csv(summary_gen_path)
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\IO_AUC_mean_SS.csv'  # summary_general
        if os.path.isfile(summary_gen_path) * skip:
            print('AUC-mean already calculated  -  skipping .. ')
        else:
            df = IOf_across.save_AUC_connection_SS(con_trial, EEG_resp, auc_summary)
            df = ls.adding_area(df, self.lbls, pair=1)
            df = ls.adding_hemisphere(df, self.lbls)
            #  df = ls.adding_distance_infalted(df, self.lbls, self.subj)
            df.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake



def start_subj(subj, cluster_method='similarity'):
    print(subj + ' -- START --')
    run_main = main(subj)
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    # load data
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    # todo: make clean
    con_trial = pd.read_csv(file_con)
    con_trial = bf.add_sleepstate(con_trial)
    con_trial['Num'] = con_trial['Num'].astype('int')
    h5_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    print('loading h5')
    EEG_resp = h5py.File(h5_file)
    EEG_resp = EEG_resp['EEG_resp']
    run_main.get_AUC_sleepstate_mean(con_trial, EEG_resp,0)
    #run_main.get_AUC_mean( con_trial, EEG_resp, 1)
    # add delay
    #run_main.add_delay(con_trial, EEG_resp)
    #run_main.get_AUC_sleepstate(con_trial, EEG_resp, 1)
    print(subj + ' ----- DONE')


sig = 0

subjs = ["EL015", "EL016", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL028","EL004","EL005", "EL014","EL010", "EL011", "EL012", "EL013"]
subjs = ["EL004","EL005","EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL020", "EL028"]
for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    start_subj(subj, 'similarity')


print('Done')
