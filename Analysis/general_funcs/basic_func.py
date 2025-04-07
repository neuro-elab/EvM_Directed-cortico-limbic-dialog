import numpy as np

import h5py
import scipy.fftpack
from general_funcs import load_summary as ls
import scipy.io as sio

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


####GENERAL functions fo read / load / prepare data

def read_mat(filename, dataname):
    try:  # open the score file of the first subfolder
        matfile = h5py.File(filename, 'r')[dataname]
        matfile = matfile[()].T
    except OSError:
        matfile = scipy.io.loadmat(filename)[dataname]
    return matfile


def get_Stim_chans(stimlist, lbls):
    labels_all = lbls.label.values
    labels_clinic = lbls.Clinic.values

    coord_all = np.array([lbls.x.values, lbls.y.values, lbls.z.values]).T
    # get stimulation channels directly from stimlist
    StimChanSM = np.unique(stimlist.ChanP)

    ChanN = np.zeros((len(StimChanSM),))
    StimChans = []  # np.zeros((len(stim_chan)))
    StimChansC = []  # np.zeros((len(stim_chan)))
    StimChanIx = []  # np.zeros((len(stim_chan)))
    i = 0
    while i < len(StimChanSM):
        ChanN[i] = np.median(stimlist[stimlist.ChanP == StimChanSM[i]].ChanN)
        if ((np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (np.array(lbls.ChanN_SM.values) == ChanN[i])).any():
            # StimChans.append(labels_SM[(np.array(labels.chan_num.values)==stim_chan[i,0])][0])
            StimChans.append(labels_all[(np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (
                    np.array(lbls.ChanN_SM.values) == ChanN[i])][0])
            StimChansC.append(labels_clinic[(np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (
                    np.array(lbls.ChanN_SM.values) == ChanN[i])][0])
            StimChanIx.append(
                lbls[(np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (np.array(lbls.ChanN_SM.values) == ChanN[i])][
                    'Num'].values[0] - 1)
            i = i + 1
        else:
            StimChanSM = np.delete(StimChanSM, i, 0)

    stimlist = stimlist[np.isin(stimlist.ChanP, StimChanSM)]

    labels_region = ls.elab2regions(lbls.Area.values)
    return labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist


def check_stim_labels(c, labels_all):
    import re
    selected_label = labels_all[c]
    # Identify the middle hyphen to split the label
    hyphens = [i for i, char in enumerate(selected_label) if char == '-']
    middle_hyphen = hyphens[len(hyphens) // 2]

    # Split the label into two sub-labels
    sub_label1 = selected_label[:middle_hyphen]
    sub_label2 = selected_label[middle_hyphen + 1:]

    # Prepare regex patterns for exact match
    pattern1 = r'\b' + re.escape(sub_label1) + r'(\b|-)'
    pattern2 = r'\b' + re.escape(sub_label2) + r'(\b|-)'

    # List to hold indices of matching labels
    matching_indices = []

    # Iterate over all labels
    for ix, label in enumerate(labels_all):
        # Check if the label contains either of the two sub-labels
        if re.search(pattern1, label) or re.search(pattern2, label):
            matching_indices.append(ix)

    return matching_indices


def check_inStimChan_C(c_s, sc_s, labels_all):
    rr = np.zeros((len(c_s), len(sc_s)))
    for j in range(len(c_s)):
        c = c_s[j]
        lb = labels_all[c]
        for i in range(len(sc_s)):
            sc = np.int64(sc_s[i])
            stim_lb = labels_all[sc]
            t = '-'
            ix = [pos for pos, char in enumerate(lb) if char == t]
            if len(ix) == 5:
                ix = np.int64(ix[3])
            elif len(ix) > 1:
                ix = np.int64(ix[1])
            else:
                ix = np.int64(ix[0])
            chan1 = lb[:ix]
            chan2 = lb[ix + 1:]
            r = 0
            if stim_lb.find(chan1) != -1:
                rr[j, i] = 1
            elif stim_lb.find(chan2) != -1:
                rr[j, i] = 1

        # print(stim_lb)
    return rr


def add_sleepstate(con_trial):
    if not 'SleepState' in con_trial:
        con_trial.insert(6, 'SleepState', 'Wake')
    if not 'Ictal' in con_trial:
        con_trial.insert(6, 'Ictal', 0)
    con_trial.loc[(con_trial.SleepState == 'W'), 'SleepState'] = 'Wake'
    con_trial.loc[(con_trial.Sleep == 0), 'SleepState'] = 'Wake'
    con_trial.loc[(con_trial.Sleep > 1) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 1), 'SleepState'] = 'NREM1'
    con_trial.loc[(con_trial.Sleep == 6), 'SleepState'] = 'SZ'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
    con_trial.loc[(con_trial.Sleep == 5), 'SleepState'] = 'Unknown'
    con_trial.loc[(con_trial.Ictal != 0), 'SleepState'] = 'SZ'
    return con_trial


def check_inStimChan(c, sc_s, labels_all):
    rr = np.zeros((len(sc_s),))
    lb = labels_all[c]
    # print(lb)
    for i in range(len(sc_s)):
        sc = np.int64(sc_s[i])
        stim_lb = labels_all[sc]
        t = '-'
        ix = [pos for pos, char in enumerate(lb) if char == t]
        if len(ix) > 3:
            ix = np.int64(ix[2])
        elif len(ix) > 1:
            ix = np.int64(ix[1])
        else:
            ix = np.int64(ix[0])
        chan1 = lb[:ix]
        chan2 = lb[ix + 1:]
        r = 0
        if stim_lb.find(chan1) != -1:
            rr[i] = 1
        elif stim_lb.find(chan2) != -1:
            rr[i] = 1

        # print(stim_lb)
    return rr


def SM2IX(SM, StimChanNums, StimChanIx):
    # SM: stim channel in SM number
    # StimChanNums: all number of stim channels in SM
    # StimChanIx: all stim channels in all channles environment
    ChanIx = np.zeros_like(SM)
    for i in range(len(SM)):
        ChanIx[i] = StimChanIx[np.where(StimChanNums == SM[i])]
    return ChanIx
