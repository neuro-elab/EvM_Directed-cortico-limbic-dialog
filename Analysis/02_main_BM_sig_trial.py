import sys
import os

# ----------------------------- #
# Set up paths for local imports
# ----------------------------- #

# Define paths to custom function folders
gf_dir = os.path.join(os.path.dirname(__file__), 'general_funcs')
st_dir = os.path.join(os.path.dirname(__file__), 'sig_trial')

# Add those paths to the system path to allow imports
sys.path.insert(0, gf_dir)
sys.path.insert(0, st_dir)

from sig_trial import run_sig_con


# ----------------------------- #
# Define global paths
# ----------------------------- #


## todo: create a globals text file which is read to define sub_path (general subjects path, and analysis path)
# path were general information of subjects are stored
sub_path  ='/Volumes/vellen/PhD/EL_experiment'
path_analysis = '/Volumes/vellen/PhD/EL_experiment/Analysis'

def run_all(subj):
    """
    Run the complete significance-based connectivity analysis pipeline for one subject.

    Steps:
    1. Compute ground truths (GT) for each channel pair using k-means cluster centers.
       Compare real single trials against surrogate data.
    2. Label trials as significant or not based on statistical thresholds.

    Parameters:
        subj (str): Subject ID (e.g., "EL010")
    """

    # Step 1: Compute ground truth (GT) and compare real trials to surrogate CCs
    run_sig_con.start_subj_GT(
        subj,
        folder='BrainMapping',
        cond_folder='CR',
        cluster_method='similarity',
        skip_GT=True,  # Set to False if GT should be recomputed
        skip_surr=True,  # Set to False if surrogate trials should be recomputed
        skip_summary=True,  # Set to False to regenerate GT summary
        trial_sig_labeling=True  # Runs trial-by-trial significance labeling
    )

    # Step 2: Perform FDR-based statistical testing for trial significance
    run_sig_con.trial_significance(
        subj,
        folder='BrainMapping',
        cond_folder='CR',
        fdr=False,  # Set to True to apply FDR correction
        p=0.05  # Significance threshold
    )


# ----------------------------- #
# Run the analysis for all subjects
# ----------------------------- #

def main():
    """
    Main entry point of the script.
    Runs the pipeline for a list of predefined subjects.
    """
    subjs = [
        "EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016",
        "EL019", "EL020", "EL021", "EL022", "EL024", "EL026", "EL027", "EL028"
    ]

    for subj in subjs:
        print(f"\nRunning pipeline for subject {subj}")
        run_all(subj)

    print('\nAll subjects processed successfully!')


# ----------------------------- #
# Script execution
# ----------------------------- #

if __name__ == '__main__':
    main()
