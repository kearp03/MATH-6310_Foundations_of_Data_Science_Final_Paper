import numpy as np
from pathlib import Path
from main import get_data
from helper_functions import read_pickle

parent_directory = Path(__file__).resolve().parent.parent

april_save_directory = parent_directory / 'April_Files'
may_save_directory = parent_directory / 'May_Files'
june_save_directory = parent_directory / 'June_Files'
all_save_directory = parent_directory / 'All_Months_Files'

# Get each dataset's data tensor and save directories
april_data_tensor, april_save_directory = get_data('april')
may_data_tensor, may_save_directory = get_data('may')
june_data_tensor, june_save_directory = get_data('june')
all_data_tensor, all_save_directory = get_data('all')

# Read in all data
april_ranks = np.array(read_pickle(april_save_directory / 'ranks.pkl'))
april_errors_SVD = np.array(read_pickle(april_save_directory / 'fro_errors_SVD.pkl'))
april_elem_stored_SVD = np.array(read_pickle(april_save_directory / 'elem_stored_SVD.pkl')) / april_data_tensor.size
april_errors_HOSVD = np.array(read_pickle(april_save_directory / 'fro_errors_HOSVD.pkl'))
april_elem_stored_HOSVD = np.array(read_pickle(april_save_directory / 'elem_stored_HOSVD.pkl')) / april_data_tensor.size

may_ranks = np.array(read_pickle(may_save_directory / 'ranks.pkl'))
may_errors_SVD = np.array(read_pickle(may_save_directory / 'fro_errors_SVD.pkl'))
may_elem_stored_SVD = np.array(read_pickle(may_save_directory / 'elem_stored_SVD.pkl')) / may_data_tensor.size
may_errors_HOSVD = np.array(read_pickle(may_save_directory / 'fro_errors_HOSVD.pkl'))
may_elem_stored_HOSVD = np.array(read_pickle(may_save_directory / 'elem_stored_HOSVD.pkl')) / may_data_tensor.size

june_ranks = np.array(read_pickle(june_save_directory / 'ranks.pkl'))
june_errors_SVD = np.array(read_pickle(june_save_directory / 'fro_errors_SVD.pkl'))
june_elem_stored_SVD = np.array(read_pickle(june_save_directory / 'elem_stored_SVD.pkl')) / june_data_tensor.size
june_errors_HOSVD = np.array(read_pickle(june_save_directory / 'fro_errors_HOSVD.pkl'))
june_elem_stored_HOSVD = np.array(read_pickle(june_save_directory / 'elem_stored_HOSVD.pkl')) / june_data_tensor.size

all_ranks = np.array(read_pickle(all_save_directory / 'ranks.pkl'))
all_errors_SVD = np.array(read_pickle(all_save_directory / 'fro_errors_SVD.pkl'))
all_elem_stored_SVD = np.array(read_pickle(all_save_directory / 'elem_stored_SVD.pkl')) / all_data_tensor.size
all_errors_HOSVD = np.array(read_pickle(all_save_directory / 'fro_errors_HOSVD.pkl'))
all_elem_stored_HOSVD = np.array(read_pickle(all_save_directory / 'elem_stored_HOSVD.pkl')) / all_data_tensor.size

print('April SVD')
print(april_errors_SVD)
print(april_elem_stored_SVD)
print('April HOSVD')
print(april_errors_HOSVD)
print(april_elem_stored_HOSVD)

print('\nMay SVD')
print(may_errors_SVD)
print(may_elem_stored_SVD)
print('May HOSVD')
print(may_errors_HOSVD)
print(may_elem_stored_HOSVD)

print('\nJune SVD')
print(june_errors_SVD)
print(june_elem_stored_SVD)
print('June HOSVD')
print(june_errors_HOSVD)
print(june_elem_stored_HOSVD)

print('\nAll SVD')
print(all_errors_SVD)
print(all_elem_stored_SVD)
print('All HOSVD')
print(all_errors_HOSVD)
print(all_elem_stored_HOSVD)
