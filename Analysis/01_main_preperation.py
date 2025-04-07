from Preperation import (
    save_hypnogram,
    start_cut_resp,
    start_BM_blocks as BM_blocks,
    start_IO_blocks as IO_blocks,
    concat,
)


# Base path to subject data
sub_path = 'X:\\4 e-Lab\\'


def run_all(subj, prots=['BM', 'IO'], cut=True, compute_trials=True, generate_hypnogram=False, concatenate=True):
    """
    Process and analyze preprocessed EEG data:
    - Cuts data into epochs (-1, 3s)
    - Calculates measurements (e.g. single trial line-length) per connection (stim x response)
    - Saves hypnograms (optional)
    - Concatenates responses into one file (optional)

    Parameters:
        subj (str): Subject ID (e.g. "EL010").
        prots (list): Protocols to process ('BM' and/or 'IO').
        cut (bool): Whether to cut EEG into epochs.
        compute_trials (bool): Whether to calculate LL/con_trial data.
        generate_hypnogram (bool): Whether to compute hypnogram from stim list.
        concatenate (bool): Whether to concatenate response data into one file.
    """

    # Map protocol name to folder
    protocol_folders = {'BM': 'BrainMapping', 'IO': 'InputOutput'}

    # Step 1: Cut data into epochs
    if cut:
        start_cut_resp.compute_cut(subj, skip_exist=1, prots=prots)

    # Step 2: Compute trial-level LL data per stimulation block
    if compute_trials:
        if 'BM' in prots:
            BM_blocks.cal_con_trial(subj, cond_folder='CR', skip_single=1)
        if 'IO' in prots:
            IO_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=1)

    # Step 3: Generate hypnogram based on stimulation timing (if enabled)
    if generate_hypnogram:
        folders = [protocol_folders[p] for p in prots if p in protocol_folders]
        save_hypnogram.run_main(subj, run_plot=0, run_save=1, folders=folders)

        # Step 4: Concatenate all epoched response data into one h5py file
        if concatenate:
            start_cut_resp.compute_list_update(subj, prots=prots)
            for prot in prots:
                folder = protocol_folders.get(prot)
                if folder:
                    concat.concat_resp_condition(subj, folder=folder, cond_folder='CR', EEG=0, skip=0)


subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL016", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL028"] # 119, "EL026", "EL027",

for subj in subjs:
    run_all(
        subj,
        prots=['BM', 'IO'],
        cut=False,
        compute_trials=True,
        generate_hypnogram=False,
        concatenate=False
    )

print('DONE ALL')

