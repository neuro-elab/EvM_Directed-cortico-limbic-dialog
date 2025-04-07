import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from glob import glob
import ntpath
import h5py
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, cohen_kappa_score

color_elab = np.zeros((4, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])
color_elab[3, :] = np.array([1, 0.574, 0])

root = Tk()
root.withdraw()
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10
})


def convert_corteo_score(df):
    # print(np.unique(df.score_string))
    score_map = {
        'QWAKE': 0, 'AWAKE': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'ICTAL': 6, 'OTHER': 5, 'ARTEFACT': 8
    }
    df['score'] = df['score_string'].map(score_map)
    return df


def align_corteo2list(shift, sleep_grades, stimlist_hypno):
    sleep_grades.Timestamp = sleep_grades.Timestamp - shift
    # Calculate end time of each score period
    sleep_grades['End'] = sleep_grades['Timestamp'] + sleep_grades['Duration']
    sleep_grades = convert_corteo_score(sleep_grades)

    # Convert Time in stimlist_hypno to datetime for comparison
    stimlist_hypno['Time'] = pd.to_datetime(stimlist_hypno['Time'])

    # Iterate over each row in stimlist_hypno to find the corresponding score
    for index, row in stimlist_hypno.iterrows():
        measurement_time = row['Time']
        # Find the corresponding score
        score_row = sleep_grades[(sleep_grades['Timestamp'] <= measurement_time.timestamp()) &
                                 (sleep_grades['End'] >= measurement_time.timestamp())]
        if not score_row.empty:
            stimlist_hypno.at[index, 'score_clinic'] = score_row.iloc[0]['score']
        if measurement_time.timestamp() > np.max(sleep_grades['End']):
            break
    return sleep_grades, stimlist_hypno


def add_clinic_score(subj, stimlist_hypno):
    stimlist_hypno['score_clinic'] = np.nan

    timeshift_file = os.path.join(sub_path, 'Patients', subj, 'Data_Raw', 'time_shift.xlsx')
    if os.path.isfile(timeshift_file):
        timeshift = pd.read_excel(timeshift_file)
        timeshift['diff_seconds'] = timeshift['diff'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        shift = timeshift['diff_seconds'].values[0]
    else:
        shift = 0

        # Now, timeshift['diff_seconds'] will contain the duration

    path_output = os.path.join(sub_path, 'Patients', subj, 'Data_Raw', 'EL_experiment', 'h5_conversion')
    h5_files = glob.glob(os.path.join(path_output, subj + '_*.h5'))

    for file in h5_files:
        with h5py.File(file, 'r') as h5_file:
            if 'sleep_grades' in h5_file:
                sleepgrades = h5_file['sleep_grades']
                sleepgradesTexts = sleepgrades['text'].asstr()[()]
                sleepgradesTimes = sleepgrades['time'][()]
                sleepgradesDurations = sleepgrades['duration'][()]

                if len(list(sleepgrades.keys())) == 3:  # add unix timestamp
                    raw = h5_file['traces/raw']
                    startTimestamp = raw.attrs['start_timestamp']
                    sleepgradesTS = sleepgradesTimes + startTimestamp
                else:
                    sleepgradesTS = sleepgrades[list(sleepgrades.keys())[3]][()]

                sleep_grades = pd.DataFrame({
                    'Onset': sleepgradesTimes,
                    'Timestamp': sleepgradesTS,
                    'Duration': sleepgradesDurations,
                    'score_string': sleepgradesTexts
                })
                sleep_grades, stimlist_hypno = align_corteo2list(shift, sleep_grades, stimlist_hypno)

            elif "CT" in os.path.basename(file):  # check whether CT is in filename
                raw = h5_file['traces/raw']
                startTimestamp = raw.attrs['start_timestamp']
                meta = h5_file['meta']
                duration = meta.attrs['duration']
                sleep_grades = pd.DataFrame({
                    'Onset': [0],
                    'Timestamp': [startTimestamp],
                    'Duration': [duration],
                    'score_string': ['AWAKE']
                })
                sleep_grades, stimlist_hypno = align_corteo2list(shift, sleep_grades, stimlist_hypno)

    return stimlist_hypno


def update_contrial_single(subj, cond_folder='CR', folder='BrainMapping'):
    # for subj in ["EL011"]:  # "EL004","EL005","EL008",EL004", "EL005", "EL008", "EL010
    # cwd = os.getcwd()
    print(f'Performing calculations on {subj}')

    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient = 'T:\EL_experiment\Patients\\' + subj + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj

    # get labels
    if cond_folder == 'Ph':
        files_list = glob(path_patient + '/Analysis/' + folder + '/data/Stim_list_*Ph*')
    elif cond_folder == 'CR':
        files_list = glob(path_patient + '/Analysis/' + folder + '/data/Stim_list_*CR*')
    else:
        print('not sure which protocol')
        return

    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'

    for l in range(0, len(files_list)):
        print('loading ' + files_list[l][-11:-4], end='\r')
        stimlist = pd.read_csv(files_list[l])
        if ('StimNum' in stimlist.columns):
            stimlist = stimlist.drop(columns='StimNum')
        stimlist.insert(5, 'StimNum', np.arange(len(stimlist)))

        # con_trial_block = BMf.LL_BM_cond(EEG_resp, stimlist, 'h', bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx)
        block_l = files_list[l][-11:-4]
        file = path_patient + '/Analysis/' + folder + '/' + cond_folder + '/data/con_trial_' + block_l + '.csv'
        con_trial_block = pd.read_csv(file)
        sleep_list = stimlist[['StimNum', 'sleep']].values
        for s in range(len(sleep_list)):
            con_trial_block.loc[con_trial_block.Num_block == sleep_list[s, 0], 'Sleep'] = sleep_list[s, 1]
        con_trial_block.to_csv(file, index=False, header=True)
        if l == 0:
            con_trial = con_trial_block
        else:
            con_trial = pd.concat([con_trial, con_trial_block])

    con_trial.to_csv(file_con, index=False, header=True)
    print(subj + ' ---- DONE Sleep Update------ ')


def update_sleep(subj, prot='BrainMapping', cond_folder='CR'):
    # updates con_trial_all based on  stimlist_hypnogram
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    file_con = path_patient_analysis + '\\' + prot + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    print('loading con_trial')
    con_trial = pd.read_csv(file_con)
    # load hypnogram
    file_hypno = path_patient_analysis + '\\stimlist_hypnogram.csv'  # path_patient + '/Analysis/stimlist_hypnogram.csv'
    if os.path.isfile(file_hypno):
        stimlist_hypno = pd.read_csv(file_hypno)
        stimlist_hypno.loc[~np.isnan(stimlist_hypno.score_clinic), 'sleep'] = stimlist_hypno.loc[
            ~np.isnan(stimlist_hypno.score_clinic), 'score_clinic']
        stimlist_hypno.loc[(stimlist_hypno.sleep == 9), 'sleep'] = 0
        for ss in np.arange(5):
            stimNum = stimlist_hypno.loc[(stimlist_hypno.sleep == ss) & (stimlist_hypno.Prot == prot), 'StimNum']
            con_trial.loc[np.isin(con_trial.Num, stimNum), 'Sleep'] = ss

    if 'SleepState' not in con_trial:
        con_trial.insert(5, 'SleepState', 'Wake')
    con_trial.loc[(con_trial.Sleep > 1) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 1), 'SleepState'] = 'NREM1'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
    con_trial.loc[(con_trial.Sleep == 6), 'SleepState'] = 'SZ'
    con_trial.to_csv(file_con, index=False, header=True)  # return con_trial
    print('con_trial updated')


def update_stimlist(subj, folder='InputOutput', cond_folder='CR'):
    print('is start_cut_resp updated?')
    # concatenates updated csv - stimlist (Single blocks) of one specific protocol
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    files = glob.glob(os.path.join(path_patient_analysis, folder, 'data', 'Stim_list_*' + cond_folder + '*'))
    files = np.sort(files)
    # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
    stimlist = []
    EEG_resp = []
    conds = np.empty((len(files),), dtype=object)
    if len(files) > 0:
        for p in range(len(files)):
            file = files[p]
            # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
            conds[p] = cond
            print(str(p + 1) + '/' + str(len(files)) + ' -- All_resps_' + file[-11:-4])
            stim_table = pd.read_csv(file)
            stim_table['type'] = cond
            if len(stimlist) == 0:
                stimlist = stim_table
            else:
                stimlist = pd.concat([stimlist, stim_table])
        stimlist = stimlist.fillna(0)
        stimlist = stimlist.reset_index(drop=True)
        stimlist['Time'] = pd.to_datetime(stimlist['date'].astype('int'), format='%Y%m%d') + pd.to_timedelta(
            stimlist['h'], unit='H') + \
                           pd.to_timedelta(stimlist['min'], unit='Min') + \
                           pd.to_timedelta(stimlist['s'], unit='Sec')
        col_drop = ["StimNum", 'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
        for d in range(len(col_drop)):
            if (col_drop[d] in stimlist.columns):
                stimlist = stimlist.drop(columns=col_drop[d])
        stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)
        stimlist.to_csv(
            path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\stimlist_' + cond_folder + '.csv',
            index=False,
            header=True)  # scat_plot
        print('data stored')


update = 1


def run_main(subj, update_list=0, update_contrial=0, folders=['BrainMapping', 'InputOutput', 'PairedPulse'], plot=True):
    print(subj)

    if update_list:
        for f in folders:  # 'BrainMapping','InputOutput','PairedPulse'
            update_stimlist(subj, folder=f, cond_folder='CR')

    file_hypno = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj + '\\stimlist_hypnogram.csv'
    # file_hypno_all = 'C:\\Users\i0328442\Desktop\hypnograms\\'+subj+'_stimlist_hypnogram.csv'
    file_hypno_fig = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj + '\\stimlist_hypnogram'  # .svg'

    stimlist_hypno = []
    prots = ['BrainMapping', 'PairedPulse', 'InputOutput']
    for p in prots:  # reads all stimlist_CR for all the protocols
        file = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj + '\\' + p + '\\CR\\data\\stimlist_CR.csv'
        if os.path.isfile(file):
            stimlist = pd.read_csv(file)
            if 'condition' in stimlist:
                stimlist = stimlist[stimlist.condition == 0]
            stimlist.insert(0, 'Prot', p)
            if len(stimlist_hypno) == 0:
                stimlist_hypno = stimlist
            else:
                stimlist_hypno = pd.concat([stimlist_hypno, stimlist])
    if not "sleep" in stimlist_hypno:
        stimlist_hypno.insert(5, 'sleep', 0)
    stimlist_hypno = stimlist_hypno.sort_values(by=['Time'])
    # stimlist_hypno.insert(5, 's', np.random.randint(0, high=59, size=(len(stimlist_hypno),), dtype=int))
    stimlist_hypno = stimlist_hypno[
        ['Time', 'Prot', 'stim_block', 'StimNum', 'ChanP', 'Int_prob', 'sleep']]  # 's'
    stimlist_hypno = stimlist_hypno.reset_index(drop=True)
    stimlist_hypno.insert(0, 'ix', np.arange(len(stimlist_hypno)))
    stimlist_hypno.insert(0, 'ix_h', pd.to_datetime(stimlist_hypno['Time']).dt.hour + pd.to_datetime(
        stimlist_hypno['Time']).dt.minute / 60 + pd.to_datetime(stimlist_hypno['Time']).dt.second / 3600)
    dates = np.unique(pd.to_datetime(stimlist_hypno['Time']).dt.date)
    d0 = 0
    for date in dates:
        stimlist_hypno.loc[(pd.to_datetime(stimlist_hypno['Time']).dt.date == date), 'ix_h'] = stimlist_hypno.loc[(
                                                                                                                          pd.to_datetime(
                                                                                                                              stimlist_hypno[
                                                                                                                                  'Time']).dt.date == date), 'ix_h'] + d0
        d0 = d0 + 24
    stimlist_hypno = add_clinic_score(subj, stimlist_hypno)
    stimlist_hypno.to_csv(file_hypno, index=False, header=True)  #
    # stimlist_hypno.to_csv(file_hypno_all, index=False, header=True)  #
    stimlist_hypno.loc[stimlist_hypno.sleep > 4, 'sleep'] = np.nan
    stimlist_hypno.loc[stimlist_hypno.score_clinic > 4, 'score_clinic'] = np.nan

    # Plot by StimNum
    file_hypno_fig = sub_path + '\\EvM\Projects\EL_experiment\Analysis\\Supp_figures\Sleep\\hypnograms\\' + subj + '_stimlist_hypnogram'  # .svg'
    plot_hyp_stimnum(subj, stimlist_hypno, file_hypno_fig)
    # Plot by Time (30s windows)
    file_hypno_fig = sub_path + '\\EvM\Projects\EL_experiment\Analysis\\Supp_figures\Sleep\\hypnograms\\' + subj + '_stimlist_hypnogram'  # .svg'
    plot_hyp_time(subj, stimlist_hypno, file_hypno_fig)
    if update_contrial:
        for f in folders:  #
            update_sleep(subj, prot=f, cond_folder='CR')


def plot_hyp_stimnum(subj, stimlist_hypno, file_hypno_fig):
    valid_scores = stimlist_hypno.dropna(subset=['score_clinic', 'sleep'])
    # Calculating Cohen's Kappa for the valid rows
    kappa = cohen_kappa_score(valid_scores['score_clinic'].astype(int), valid_scores['sleep'].astype(int))

    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(1, 2, width_ratios=[2, 1])  # Adjust the ratio here as needed
    # First subplot: Scores comparison
    ax1 = fig.add_subplot(gs[0])
    plot_scores(ax1, stimlist_hypno, subj, kappa, delta='Stim')
    # Second subplot: Confusion matrix
    ax2 = fig.add_subplot(gs[1])
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(valid_scores['sleep'].astype(int), valid_scores['score_clinic'].astype(int), labels=range(5))
    sns.heatmap(cm, annot=True, ax=ax2, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Scorer 2')
    ax2.set_ylabel('Scorer 1')
    plt.tight_layout()
    plt.savefig(file_hypno_fig + '_stimNum.pdf')
    plt.close()


def robust_mode(x):
    if not x.empty:
        mode_result = x.mode()
        if not mode_result.empty:
            return mode_result[0]
    return None


def plot_hyp_time(subj, stimlist_hypno, file_hypno_fig):
    df = stimlist_hypno.dropna(subset=['score_clinic', 'sleep']).reset_index(drop=True)
    df_all = stimlist_hypno.reset_index(drop=True)

    df['Time'] = pd.to_datetime(df['Time'])
    df_all['Time'] = pd.to_datetime(df_all['Time'])

    df.set_index('Time', inplace=True)
    df_all.set_index('Time', inplace=True)

    result_df = df.resample('30S').agg(robust_mode)
    df_all = df_all.resample('30S').agg(robust_mode)

    result_df.reset_index(inplace=True)
    df_all.reset_index(inplace=True)
    result_df = result_df.dropna(subset=['score_clinic', 'sleep']).reset_index(drop=True)

    kappa = cohen_kappa_score(result_df['score_clinic'].astype(int), result_df['sleep'].astype(int))

    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(1, 2, width_ratios=[2, 1])  # Adjust the ratio here as needed
    # First subplot: Scores comparison
    ax1 = fig.add_subplot(gs[0])
    plot_scores(ax1, df_all, subj, kappa)  # Ensure result_df is defined as per your setup

    # Second subplot: Confusion matrix
    ax2 = fig.add_subplot(gs[1])
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(result_df['sleep'].astype(int), result_df['score_clinic'].astype(int), labels=range(5))
    sns.heatmap(cm, annot=True, ax=ax2, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Scorer 2')
    ax2.set_ylabel('Scorer 1')
    plt.tight_layout()
    # plt.savefig(file_hypno_fig + '_Time.svg')
    plt.savefig(file_hypno_fig + '_Time.pdf')
    plt.close()


def plot_scores(ax, data, subj, kappa, delta='Time'):
    if delta == 'Time':
        # Ensure 'Time' is in datetime format and set as index
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)

        # Plotting both scores on the given ax
        ax.plot(data.index, data['sleep'], c=color_elab[2, :], linewidth=0.75, alpha=0.95,
                label='Scorer 1')
        ax.plot(data.index, data['score_clinic'] - 0.1, c='k', linewidth=0.75, alpha=0.8,
                label='Scorer 2')
        ax.set_xlabel('Time')
    else:
        ax.plot(data.ix, data.sleep, c=color_elab[2, :], linewidth=0.75, alpha=0.95,
                label='Scorer 1')
        ax.plot(data.ix, data.score_clinic - 0.1, c='k', linewidth=0.75, alpha=0.8,
                label='Scorer 2')
        ax.set_xlabel('Stimulation Number')
    # Enhancing the plot
    ax.set_title('Clinic and Sleep Scores Over Time')
    ax.set_ylabel('Scores')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax.set_ylim([-1, 5])
    ax.invert_yaxis()

    ax.set_title(f'{subj} -- Kappa {kappa:0.2f}')
    ax.legend()
    # ax.grid(True)


def save_agreement_across(subjs):
    from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
    # Initialize an empty DataFrame with the desired columns
    df = pd.DataFrame(columns=['subj', 'Kappa', 'Accuracy', 'Acc. Wake', 'Acc. NREM', 'Acc. REM'])

    for ix_subj, subj in enumerate(subjs):
        path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', subj)
        file = os.path.join(path_patient_analysis, 'stimlist_hypnogram.csv')
        stimlist_hypno = pd.read_csv(file)
        # remove times where artefacts or seizures are stored(Paivi did it in scoring, I did it in epitome as bad times)
        stimlist_hypno.loc[stimlist_hypno.sleep > 4, 'sleep'] = np.nan
        stimlist_hypno.loc[stimlist_hypno.score_clinic > 4, 'score_clinic'] = np.nan
        valid_scores = stimlist_hypno.dropna(subset=['score_clinic', 'sleep']).reset_index(drop=True)
        # acc = accuracy_score(valid_scores['score_clinic'].astype(int), valid_scores['sleep'].astype(int))
        # kappa = cohen_kappa_score(valid_scores['score_clinic'].astype(int), valid_scores['sleep'].astype(int))

        # Convert 'Time' column to datetime
        valid_scores['Time'] = pd.to_datetime(valid_scores['Time'])

        # Set 'Time' as the index
        valid_scores.set_index('Time', inplace=True)

        # Resample and aggregate
        result_df = valid_scores.resample('30S').agg(lambda x: x.mode()[0] if not x.empty else None)

        # Reset index if you want 'Time' back as a column
        result_df.reset_index(inplace=True)
        result_df = result_df.dropna(subset=['score_clinic', 'sleep']).reset_index(drop=True)
        kappa = cohen_kappa_score(result_df['score_clinic'].astype(int), result_df['sleep'].astype(int))
        acc = accuracy_score(result_df['score_clinic'].astype(int), result_df['sleep'].astype(int))

        valid_scores.loc[valid_scores.sleep == 2, 'sleep'] = 3
        valid_scores.loc[valid_scores.score_clinic == 2, 'score_clinic'] = 3
        acc_sub = np.zeros((4,))
        for ix, val in enumerate([0, 1, 3, 4]):
            score1 = valid_scores.sleep.values.copy()
            score1[score1 != val] = -1
            score2 = valid_scores.score_clinic.values.copy()
            score2[score2 != val] = -1
            acc_sub[ix] = accuracy_score(score1, score2)
        # Append the new row to the DataFrame
        df = df.append({
            'subj': subj,
            'Kappa': kappa,
            'Accuracy': acc,
            'Acc. Wake': acc_sub[0],
            'Acc. NREM (N1)': acc_sub[1],
            'Acc. NREM (N2, N3)': acc_sub[2],
            'Acc. REM': acc_sub[3]
        }, ignore_index=True)
    plot_agreement_across(df)


def plot_agreement_across(df):
    metrics = ['Kappa', 'Accuracy', 'Acc. Wake', 'Acc. NREM (N2, N3)', 'Acc. REM']
    fig, axes = plt.subplots(1, len(metrics), figsize=(8, 4), sharey=True)  # Adjusted for better visualization

    # Define a color palette for the stripplot
    unique_subjects = df['subj'].nunique()
    palette = sns.color_palette('colorblind', n_colors=unique_subjects)
    df['x'] = 0
    for i, metric in enumerate(metrics):
        sns.boxplot(y=metric, data=df, ax=axes[i], color='white', linewidth=2, fliersize=0)
        sns.stripplot(x='x', y=metric, data=df, ax=axes[i], hue='subj', palette=palette, dodge=False, jitter=True,
                      alpha=1.)
        # sns.scatterplot(y=metric, data=df, ax=axes[i], hue='subj', palette=palette, s= 10)
        axes[i].set_ylim([0, 1])
        axes[i].set_ylabel('')
        axes[i].set_title(metric)

        # Hide legend for all but the last plot
        # Check if the legend was created and remove it
        legend = axes[i].get_legend()
        if legend:
            legend.remove()

    # Adjust legend to show on the right
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), title='Subject')

    plt.tight_layout()  # Adjust layout to not overlap
    filename = 'X:\\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Supp_figures\Sleep\hypnograms\\sleep_agreement.svg'
    plt.savefig(filename)
    filename = 'X:\\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Supp_figures\Sleep\hypnograms\\sleep_agreement.pdf'
    plt.savefig(filename)
    plt.show()
