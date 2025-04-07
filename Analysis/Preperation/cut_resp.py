import os
import numpy as np
import mne
import h5py
import pandas as pd
import sys
from tkinter import *
import ntpath
from pandas import read_excel, ExcelWriter
root = Tk()
root.withdraw()
import copy
from scipy.io import savemat
import scipy
import platform
from glob import glob

# import SM_converter as SMC
cwd = os.getcwd()


class main:
    def __init__(self, subj, path_patient, dur=np.array([-1, 3])):

        # if not os.path.exists(path_gen):
        #    path_gen = 'T:\\EL_experiment\\Patients\\' + subj  # if not in hulk check in T drive
        # path_patient = path_gen + '\Data\EL_experiment'  # path where data is stored
        path_infos = os.path.join(path_patient, 'infos')  # infos is either in Data or in general
        if not os.path.exists(path_infos):
            print('Cant find infos folder in patients folder')
            # path_infos = path_gen + '\\infos'

        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = dur  # [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_patient, 'Electrodes', subj + "_labels.xlsx"), header=0, sheet_name='BP')
        if "type" in lbls:
            lbls = lbls[lbls.type == 'SEEG']
            lbls = lbls.reset_index(drop=True)
        self.labels = lbls.label.values
        self.labels_C = lbls.Clinic.values

        self.coord_all = np.array([lbls.x.values, lbls.y.values, lbls.z.values]).T
        # only healthy channels
        # tissue  = lbls[req].Tissue.values
        self.subj = subj
        self.path_patient = path_patient
        self.path_patient_analysis = os.path.join('Y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients', subj)
        self.path_patient_analysis = os.path.join('X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients', subj)


    def cut_resp_LT(self, path, path_save):
        types = ['LTD1', 'LTD10', 'LTP50']
        folder = 'LongTermInduction'

        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000

        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-3]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            if t[-1] == '_':
                t = t[:-1]
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
            p = 0
        file_block = path_save + '\\' + folder + '/data/All_resps_' + str(p).zfill(
            2) + '_' + t + '.npy'

        if not os.path.exists(path_save + '\\' + folder):
            os.makedirs(path_save + '\\' + folder)
            os.makedirs(path_save + '\\' + folder + '\\data')

        stim_table = stim_table.drop(columns="Num", errors='ignore')
        stim_table = stim_table.reset_index(drop=True)
        stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
        if len(stim_table) > 0:
            if not os.path.exists(path_save + '\\' + folder + '/data/'):
                os.makedirs(path_save + '\\' + folder + '/data/')
            # Get bad channels
            if os.path.isfile(path + "/bad_chs.mat"):
                try:  # load bad channels
                    matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                    bad_chan = matfile[()].T
                except IOError:
                    bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
                if len(bad_chan) == 0:
                    bad_chan = np.zeros((len(self.labels), 1))
            else:
                bad_chan = np.zeros((len(self.labels), 1))
            try:
                badchans = pd.read_csv(path_save + '\\' + folder + '/data/badchan.csv')
                badchans = badchans.drop(columns=str(p), errors='ignore')
                badchans.insert(loc=1, column=str(p), value=bad_chan[:, 0])
                # new_column = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                # badchans[str(block)] = bad_chan[:, 0]
            except FileNotFoundError:
                badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(p): bad_chan[:, 0]})
            badchans.to_csv(path_save + '\\' + folder + '/data/badchan.csv', index=False,
                            header=True)  # scat_plot

            # get data
            EEG_block = np.zeros((len(self.labels), len(stim_table), self.dur_tot * self.Fs))
            EEG_block[:, :, :] = np.NaN
            # load matlab EEG
            try:
                matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
                EEGpp = matfile[()].T
            except IOError:
                EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

            print('ppEEG loaded ')
            # go through each stim trigger
            for s in range(len(stim_table)):
                trig = stim_table.TTL_DS.values[s]
                if not np.isnan(trig):
                    if np.int64(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                        EEG_block[:, s, 0:EEGpp.shape[1] - np.int64(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                                        np.int64(
                                                                                                            trig +
                                                                                                            self.dur[
                                                                                                                0, 0] * self.Fs):
                                                                                                        EEGpp.shape[
                                                                                                            1]]
                    elif np.int64(trig + self.dur[0, 0] * self.Fs) < 0:
                        EEG_block[:, s, abs(np.int64(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int64(
                            trig + self.dur[0, 1] * self.Fs)]
                    else:
                        EEG_block[:, s, :] = EEGpp[:, np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(
                            trig + self.dur[0, 1] * self.Fs)]

            np.save(file_block,
                    EEG_block)
            stim_table.to_csv(path_save + '\\' + folder + '/data/Stim_list_' + str(p).zfill(
                2) + '_' + t + '.csv', index=False,
                              header=True)  # scat_plot

    def cut_resp_IOM(self, path_pp, path_save, prot='IOM'):
        # get data
        num_stim = 20 * 60  # number of stimulation (20minutes) # hardcoded !
        # load matlab EEG
        try:
            matfile = h5py.File(path_pp + "/ppEEG.mat", 'r')['ppEEG']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(path_pp + "/ppEEG.mat")['ppEEG']
        trig = np.arange(0, self.Fs * (num_stim + 1), 500).astype('int')  # all trigger. every second
        EEG_block = np.zeros((len(self.labels), len(trig), self.dur_tot * self.Fs))
        EEG_block[:, :, :] = np.NaN
        #
        for t, s in zip(trig, np.arange(num_stim)):
            EEG_block[:, s, :] = EEGpp[:,
                                 np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(trig + self.dur[0, 1] * self.Fs)]

        # todo:
        np.save(path_save + '\\Epoch_data_' + prot + '.npy', EEG_block)



    def cut_resp_general(self, path, path_analysis, block, type, skip=1):
        # Define the folder and types based on input type
        type_mapping = {
            'PP': (['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP'], 'PairedPulse'),
            'IO': (['IO', 'Ph_IO', 'CR_IO', 'InputOutput'], 'InputOutput'),
            'BM': (['BM', 'BM_BM', 'BMCT', 'Ph_BM', 'CR_BM', 'BrainMapping'], 'BrainMapping'),
            'LT': (['LTD1', 'LTD10', 'LTP50'], 'LongTermInduction')
        }

        types, folder = type_mapping.get(type, ([], 'nofolder'))

        if not types:
            print('ERROR: no type defined (BM, IO, PP)')
            return

        # Create folder where data is stored
        path_save = os.path.join(path_analysis, folder)
        os.makedirs(os.path.join(path_save, 'data'), exist_ok=True)

        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[:5]

        if not os.path.isfile(os.path.join(path, "ppEEG.mat")):
            return

        t = filename[9:].rstrip('0123456789_')
        p = int(filename[9 + len(t):]) if filename[9 + len(t):].isdigit() else 0

        stim_table_path = os.path.join(data_path, f"{subj}_stimlist_{t}.xlsx")
        stim_table = pd.read_excel(stim_table_path, sheet_name=f'Sheet{p}' if p else 0)

        condition = t[:2]
        filename_block = os.path.join(path_save, 'data',
                                      f'All_resps_{str(block).zfill(2)}_{condition}{str(p).zfill(2)}.npy')
        short_name = f'{str(block).zfill(2)}_{condition}{str(p).zfill(2)}'

        if os.path.isfile(filename_block) and skip:
            print(f'{short_name} already exist', end='\r')
            return

        print(f'{short_name} cutting', end='\r')

        if type == 'LT':
            stim_table = stim_table[stim_table['type'].str.contains('|'.join(types))]
            stim_table = stim_table[~stim_table['type'].str.contains('Prot')]
            folder = os.path.join(folder, t)
        else:
            stim_table = stim_table[stim_table['type'].isin(types)]

        stim_table = stim_table.drop(columns="Num", errors='ignore').reset_index(drop=True)
        stim_table = stim_table[(stim_table.ChanP > 0) & (stim_table.ChanN > 0)].reset_index(drop=True)
        stim_table.insert(10, "Num", np.arange(len(stim_table)))

        if stim_table.empty:
            print('No Stimulation in this protocol')
            return

        # Get bad channels
        path_badchans = os.path.join(path_save, 'data', 'badchan.csv')
        if os.path.isfile(os.path.join(path, "bad_chs.mat")):
            try:
                matfile = h5py.File(os.path.join(path, "bad_chs.mat"), 'r')['bad_chs']
                bad_chan = matfile[()].T
            except IOError:
                bad_chan = scipy.io.loadmat(os.path.join(path, "bad_chs.mat"))['bad_chs']
            if len(bad_chan) == 0:
                bad_chan = np.zeros((len(self.labels), 1))
        else:
            bad_chan = np.zeros((len(self.labels), 1))

        try:
            badchans = pd.read_csv(path_badchans)
            badchans[str(block)] = bad_chan[:, 0]
        except FileNotFoundError:
            badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})

        badchans.to_csv(path_badchans, index=False, header=True)

        # Get data
        EEG_block = np.full((len(self.labels), len(stim_table), self.dur_tot * self.Fs), np.NaN)

        try:
            matfile = h5py.File(os.path.join(path, "ppEEG.mat"), 'r')['ppEEG']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(os.path.join(path, "ppEEG.mat"))['ppEEG']

        print('ppEEG loaded')

        # Process each stim trigger
        for s, trig in enumerate(stim_table.TTL_DS.values):
            if not np.isnan(trig):
                start_idx = int(trig + self.dur[0, 0] * self.Fs)
                end_idx = int(trig + self.dur[0, 1] * self.Fs)
                if start_idx < 0:
                    EEG_block[:, s, abs(start_idx):] = EEGpp[:, :end_idx]
                elif end_idx > EEGpp.shape[1]:
                    EEG_block[:, s, :(EEGpp.shape[1] - start_idx)] = EEGpp[:, start_idx:]
                else:
                    EEG_block[:, s, :] = EEGpp[:, start_idx:end_idx]

        np.save(filename_block, EEG_block)
        filename_stimtable = os.path.join(path_save, 'data',
                                          f'Stim_list_{str(block).zfill(2)}_{condition}{str(p).zfill(2)}.csv')
        stim_table.to_csv(filename_stimtable, index=False, header=True)

    def cut_resp(self, path, block, type, skip=1):
        ###MAIN FUNCTION
        # infos, always the same
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'BM_BM', 'BMCT', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        elif type == 'LT':
            types = ['LTD1', 'LTD10', 'LTP50']
            folder = 'LongTermInduction'
        else:
            types = []
            folder = 'nofolder'
        if len(types) > 0:
            # Patient specific
            filename = ntpath.basename(path)
            data_path = os.path.dirname(os.path.dirname(path))
            subj = filename[0:5]  # EL000
            # data
            path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            if os.path.isfile(path + "/ppEEG.mat"):
                t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
                if filename[-1].isnumeric():
                    if filename[-2].isnumeric():
                        t = filename[9:-2]
                        p = int(filename[-2:])
                    else:
                        t = filename[9:-1]
                        p = int(filename[-1])
                    if t[-1] == '_':
                        t = t[:-1]
                    stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                               sheet_name='Sheet' + str(p))  #
                else:
                    stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
                    p = 0
                condition = t[0:2]
                filename_block = self.path_patient_analysis + '\\' + folder + '/data/All_resps_' + str(block).zfill(
                    2) + '_' + condition + str(p).zfill(2) + '.npy'
                short_name = str(block).zfill(
                    2) + '_' + condition + str(p).zfill(2)
                if os.path.isfile(filename_block) * skip:
                    print(short_name + ' already exist', end='\r')
                else:
                    print(short_name + ' cutting', end='\r')
                    if type == 'LT':
                        stim_table = stim_table[stim_table['type'].str.contains('|'.join(types))]
                        # remove induction stimulations from stim list. we dont anaylze these for now..
                        stim_table = stim_table[~stim_table['type'].str.contains('Prot')]
                        folder = folder + '\\' + t
                    else:
                        stim_table = stim_table[stim_table['type'].isin(types)]
                    if not os.path.exists(self.path_patient_analysis + '\\' + folder):
                        os.makedirs(self.path_patient_analysis + '\\' + folder)
                        os.makedirs(self.path_patient_analysis + '\\' + folder + '\\data')

                    stim_table = stim_table.drop(columns="Num", errors='ignore')
                    stim_table = stim_table.reset_index(drop=True)
                    stim_table = stim_table[stim_table.ChanP > 0]
                    stim_table = stim_table[stim_table.ChanN > 0]
                    stim_table = stim_table.reset_index(drop=True)
                    stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
                    if len(stim_table) > 0:
                        if not os.path.exists(self.path_patient_analysis + '\\' + folder + '/data/'):
                            os.makedirs(self.path_patient_analysis + '\\' + folder + '/data/')
                        # Get bad channels
                        if os.path.isfile(path + "/bad_chs.mat"):
                            try:  # load bad channels
                                matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                                bad_chan = matfile[()].T
                            except IOError:
                                bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
                            if len(bad_chan) == 0:
                                bad_chan = np.zeros((len(self.labels), 1))
                        else:
                            bad_chan = np.zeros((len(self.labels), 1))
                        try:
                            badchans = pd.read_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv')
                            badchans = badchans.drop(columns=str(block), errors='ignore')
                            badchans.insert(loc=1, column=str(block), value=bad_chan[:, 0])
                            # new_column = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                            # badchans[str(block)] = bad_chan[:, 0]
                        except FileNotFoundError:
                            badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                        badchans.to_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv', index=False,
                                        header=True)  # scat_plot

                        # get data
                        EEG_block = np.zeros((len(self.labels), len(stim_table), self.dur_tot * self.Fs))
                        EEG_block[:, :, :] = np.NaN
                        # load matlab EEG
                        try:
                            matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
                            EEGpp = matfile[()].T
                        except IOError:
                            EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

                        print('ppEEG loaded ')
                        # go through each stim trigger
                        for s in range(len(stim_table)):
                            trig = stim_table.TTL_DS.values[s]
                            if not np.isnan(trig):
                                if np.int64(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                                    EEG_block[:, s,
                                    0:EEGpp.shape[1] - np.int64(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                                    np.int64(
                                                                                                        trig +
                                                                                                        self.dur[
                                                                                                            0, 0] * self.Fs):
                                                                                                    EEGpp.shape[
                                                                                                        1]]
                                elif np.int64(trig + self.dur[0, 0] * self.Fs) < 0:
                                    EEG_block[:, s, abs(np.int64(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:,
                                                                                                       0:np.int64(
                                                                                                           trig +
                                                                                                           self.dur[
                                                                                                               0, 1] * self.Fs)]
                                else:
                                    EEG_block[:, s, :] = EEGpp[:, np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(
                                        trig + self.dur[0, 1] * self.Fs)]

                        np.save(filename_block,
                                EEG_block)
                        stim_table.to_csv(
                            self.path_patient_analysis + '\\' + folder + '/data/Stim_list_' + str(block).zfill(
                                2) + '_' + condition + str(p).zfill(2) + '.csv', index=False,
                            header=True)  # scat_plot
                    else:
                        print('No Stimulation in this protocol')
        else:
            print('ERROR: no type defined (BM, IO, PP)')


    def list_update(self, path, block, type):
        # infos, always the same
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'
        if len(types) > 0:
            # Patient specific
            filename = ntpath.basename(path)
            data_path = os.path.dirname(os.path.dirname(path))
            subj = filename[0:5]  # EL000
            # data
            path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-

            if filename[-1].isnumeric():
                if filename[-2].isnumeric():
                    t = filename[9:-2]
                    p = int(filename[-2:])
                else:
                    t = filename[9:-1]
                    p = int(filename[-1])
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                           sheet_name='Sheet' + str(p))  #
            else:
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
                p = 0
            print(t)
            condition = t[0:2]
            if type == 'LT':
                stim_table = stim_table[stim_table['type'].str.contains('|'.join(types))]
            else:
                stim_table = stim_table[stim_table['type'].isin(types)]

            stim_table = stim_table.drop(columns="Num", errors='ignore')
            stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
            if len(stim_table) > 0:
                # Get bad channels
                if os.path.isfile(path + "/bad_chs.mat"):
                    try:  # load bad channels
                        matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                        bad_chan = matfile[()].T
                    except IOError:
                        bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
                    if len(bad_chan) == 0:
                        bad_chan = np.zeros((len(self.labels), 1))
                else:
                    bad_chan = np.zeros((len(self.labels), 1))
                try:
                    badchans = pd.read_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv')
                    badchans = badchans.drop(columns=str(block), errors='ignore')
                    badchans.insert(loc=1, column=str(block), value=bad_chan[:, 0])
                    # new_column = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                    # badchans[str(block)] = bad_chan[:, 0]
                except FileNotFoundError:
                    badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                badchans.to_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv', index=False,
                                header=True)  # scat_plot
                # todo: two digit number of block
                col_drop = ["StimNum", 'StimNum.1', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS',
                            'currentflow']
                for d in range(len(col_drop)):
                    if (col_drop[d] in stim_table.columns):
                        stim_table = stim_table.drop(columns=col_drop[d])
                stim_table.insert(0, "StimNum", np.arange(len(stim_table)), True)
                stim_table = stim_table.reset_index(drop=True)

                stim_table.to_csv(self.path_patient_analysis + '\\' + folder + '/data/Stim_list_' + str(block).zfill(
                    2) + '_' + condition + str(p).zfill(2) + '.csv', index=False,
                                  header=True)  # scat_plot

                print('stimlist updated')
        else:
            print('ERROR: no valid type name')

    def concat_resp(self, type):
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'

        files = glob(self.path_patient + '/Analysis/' + folder + '/data/Stim_list_*')
        files = np.sort(files)
        # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
        stimlist = []
        EEG_resp = []
        conds = np.empty((len(files),), dtype=object)
        for p in range(len(files)):
            file = files[p]
            # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
            conds[p] = cond
            k = int(ntpath.basename(file)[idxs[0]:idxs[1] + 1])
            EEG_block = np.load(self.path_patient + '/Analysis/' + folder + '/data/All_resps_' + file[-11:-4] + '.npy')
            print(str(p + 1) + '/' + str(len(files)) + ' -- All_resps_' + file[-11:-4])
            stim_table = pd.read_csv(file)
            stim_table['type'] = cond
            if len(stimlist) == 0:
                EEG_resp = EEG_block
                stimlist = stim_table
            else:
                EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)
                stimlist = pd.concat([stimlist, stim_table])
            # del EEG_block
            # os.remove(self.path_patient + '/Analysis/'+folder+'/data/All_resps_' + str(k) + '_' + cond + '.npy')
            # os.remove(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_' + cond + '.csv')
        stimlist = stimlist.drop(columns="StimNum", errors='ignore')

        # stimlist.loc[(stimlist.Condition == 'Benzo'), 'sleep']  = 5
        # stimlist.loc[(stimlist.Condition == 'Fuma'), 'sleep']   = 6
        stimlist = stimlist.fillna(0)
        stimlist = stimlist.reset_index(drop=True)
        col_drop = ["StimNum", 'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
        for d in range(len(col_drop)):
            if (col_drop[d] in stimlist.columns):
                stimlist = stimlist.drop(columns=col_drop[d])
        stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)

        np.save(self.path_patient + '/Analysis/' + folder + '/data/All_resps.npy', EEG_resp)
        stimlist.to_csv(self.path_patient + '/Analysis/' + folder + '/data/Stimlist.csv', index=False,
                        header=True)  # scat_plot
        print('data stored')
        print(self.path_patient + '/Analysis/' + folder + '/data/All_resps.npy')


    def concat_list(self, type):
        if type == 'PP':
            folder = 'PairedPulse'
        elif type == 'IO':
            folder = 'InputOutput'
        elif type == 'BM':
            folder = 'BrainMapping'
        else:
            folder = 'nofolder'

        files = glob(self.path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
        files = np.sort(files)

        if len(files)>0:# prots           = np.int64(np.arange(1, len(files) + 1))  # 43
            stimlist = []
            conds = np.empty((len(files),), dtype=object)
            for p in range(len(files)):
                file = files[p]
                # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
                idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]

                cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
                conds[p] = cond
                k = int(ntpath.basename(file)[idxs[0]:idxs[1] + 1])
                stim_table = pd.read_csv(file)
                stim_table['type'] = cond
                if len(stimlist) == 0:
                    stimlist = stim_table
                else:
                    stimlist = pd.concat([stimlist, stim_table])
                # os.remove(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_' + cond + '.csv')

            col_drop = ['StimNum', 'StimNum.1', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
            for d in range(len(col_drop)):
                if (col_drop[d] in stimlist.columns):
                    stimlist = stimlist.drop(columns=col_drop[d])
            stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)

            stimlist.to_csv(self.path_patient_analysis + '\\' + folder + '/data/Stimlist.csv', index=False,
                            header=True)  # scat_plot
            print('data stored')
            print(self.path_patient + '/Analysis/' + folder + '/data/Stimlist.csv')

    def cut_BM(self, path, block):

        # infos, always the same
        types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000
        # data
        path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))

        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-2]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
        print(t)
        condition = t[0:2]
        stim_table = stim_table[stim_table['type'].isin(types)]
        # get data
        EEG_block = np.zeros((len(self.labels), len(stim_table), self.dur_tot * self.Fs))
        EEG_block[:, :, :] = np.NaN
        try:
            matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

        print('ppEEG loaded ')
        for s in range(len(stim_table)):
            trig = stim_table.TTL_DS.values[s]
            if np.int(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                EEG_block[:, s, 0:EEGpp.shape[1] - np.int(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                              np.int(trig + self.dur[
                                                                                                  0, 0] * self.Fs):
                                                                                              EEGpp.shape[1]]
            elif np.int(trig + self.dur[0, 0] * self.Fs) < 0:
                EEG_block[:, s, abs(np.int(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int(
                    trig + self.dur[0, 1] * self.Fs)]
            else:
                EEG_block[:, s, :] = EEGpp[:,
                                     np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

        np.save(path_patient + '/Analysis/BrainMapping/data/All_resps_' + str(block) + '_' + condition + '.npy',
                EEG_block)
        stim_table.to_csv(
            path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(block) + '_' + condition + '.csv',
            index=False,
            header=True)  # scat_plot
