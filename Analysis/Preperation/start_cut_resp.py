import os
from general_funcs import freq_funcs as ff
from general_funcs import basic_func as bf
from general_funcs import LL_funcs as LLf
from Preperation  import cut_resp
import glob

cwd = os.getcwd()


def compute_cut_general(subj, path_raw, path_analysis, skip_exist=1, prots = ['BM', 'IO', 'PP']):
    print(f'Performing calculations on {subj}')
    # path_patient = 'T:\EL_experiment\Patients\\' + subj  ##'T:\EL_experiment\Patients'#os.path.join('E:\PhD\EL_experiment\Patients',subj) # path_patient    = 'E:\PhD\EL_experiment\Patients\\'+subj ##'T:\EL_experiment\Patients'
    path_patient = 'X:\\4 e-Lab\\Patients\\'+ subj
    CUT = cut_resp.main(subj, path_patient)
    paths = os.listdir(path_raw)
    n_ex = len(paths)
    k = 0

    print('Cutting responses into [1,3]s epochs ... ')
    for n in range(n_ex):
        if not paths[n][:5] == 'infos':
            path_data = os.path.join(path_raw, paths[n])
            folders = glob.glob(path_data + '\\' + subj + '_*')
            if len(folders) > 0:
                for i in range(0, len(folders)): # todo: change back to 0
                    for prot in prots:

                        CUT.cut_resp_general(folders[i], path_analysis, k + i + 1, prot,skip_exist)

                k = k + i

    print(subj + ' ---- DONE ------ ')

def compute_cut(subj, skip_exist=1, prots = ['BM', 'IO', 'PP']):
    print(f'Performing calculations on {subj}')
    # path_patient = 'T:\EL_experiment\Patients\\' + subj  ##'T:\EL_experiment\Patients'#os.path.join('E:\PhD\EL_experiment\Patients',subj) # path_patient    = 'E:\PhD\EL_experiment\Patients\\'+subj ##'T:\EL_experiment\Patients'
    path_patient = 'X:\\4 e-Lab\\Patients\\'+ subj
    CUT = cut_resp.main(subj, path_patient)
    paths = os.listdir(path_patient + '\Data\EL_experiment')
    n_ex = len(paths)
    k = 0

    print('Cutting responses into [1,3]s epochs ... ')
    for n in range(n_ex):
        if not paths[n][:5] == 'infos':
            path_data = os.path.join(path_patient, 'Data\EL_experiment', paths[n], 'data_blocks')
            folders = glob.glob(path_data + '\\' + subj + '_*')
            if len(folders) > 0:
                for i in range(0, len(folders)): # todo: change back to 0
                    for prot in prots:

                        CUT.cut_resp(folders[i], k + i + 1, prot,skip_exist)

                k = k + i

    print(subj + ' ---- DONE ------ ')


def compute_list_update(subj, prots = ['BM', 'IO', 'PP']):
    print(f'Performing calculations on {subj}')
    path_patient = 'Y:\\eLab\\Patients\\' + subj  ##'T:\EL_experiment\Patients'#os.path.join('E:\PhD\EL_experiment\Patients',subj) # path_patient    = 'E:\PhD\EL_experiment\Patients\\'+subj ##'T:\EL_experiment\Patients'
    # path_patient    = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/'+subj
    path_patient = 'X:\\4 e-Lab\\Patients\\' + subj
    CUT = cut_resp.main(subj, path_patient)
    #paths = os.listdir(path_patient + '\Data\EL_experiment')
    #n_ex = len(paths)
    #n_ex = len(paths)
    k = 0  # todo: 0

    for n in range(1):
        #path_data = os.path.join(path_patient, 'Data\EL_experiment', paths[n], 'data_blocks')
        path_data = os.path.join(path_patient, 'Data\EL_experiment', 'experiment1', 'data_blocks')
        folders = glob.glob(path_data + '\\' + subj + '_*')

        if len(folders) > 0:
            for i in range(len(folders)):
                for prot in prots:
                    CUT.list_update(folders[i], k + i + 1, prot)

            k = k + i
    for prot in prots:
        CUT.concat_list(prot)
    print(subj + ' ---- DONE ------ ')


#
# compute_cut('EL014')
# for subj in ["EL022"]:  # , "EL010"
    # compute_list_update(subj)
    # compute_cut(subj, cut_blocks=1, concat_blocks=0)  # _thread.start_new_thread(compute_list_update, (subj,))
    # compute_cut(subj, concat_blocks=0)
    # compute_list_update(subj)
# while 1:
#    time.sleep(1)

