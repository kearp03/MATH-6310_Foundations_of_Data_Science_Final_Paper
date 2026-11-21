import numpy as np
import scipy.io
from pathlib import Path
import pickle
# import SVD
# import helper_functions

if __name__ == '__main__':
    # Get directories as Paths
    current_directory = Path.cwd()
    parent_directory = current_directory.parent
    data_directory = Path()
    save_directory = Path()
    # Determine which month to analyze
    month = input('Enter month to analyze: ').lower()
    # Set directories as appropriate for each month
    match month:
        case 'april':
            data_directory = parent_directory / 'Prepared_April_2015'
            save_directory = parent_directory / 'April_Files'
        case 'may':
            data_directory = parent_directory / 'Prepared_May_2015'
            save_directory = parent_directory / 'May_Files'
        case 'june':
            data_directory = parent_directory / 'Prepared_June_2015'
            save_directory = parent_directory / 'June_Files'
        case _:
            print('Month not available')
            exit()
    # Try to read in data tensor from pickle file
    try:
        with open(save_directory / 'data_tensor.pkl', 'rb') as file:
            data_tensor: np.ndarray = pickle.load(file)
    # If file hasn't been created yet, read in Matlab files to create it
    except FileNotFoundError:
        print('Creating data tensor')
        number_of_days = sum([1 for file in data_directory.glob('SM_*.mat') if file.is_file() and file.suffix.casefold() == '.mat'.casefold()])
        
        data_tensor = np.zeros((3312,3276,number_of_days))
        
        for i, file_path in enumerate(data_directory.glob('SM_*.mat')):
            data_tensor[:,:,i] = scipy.io.loadmat(file_path)['D']
        
        with open(save_directory / 'data_tensor.pkl', 'wb') as file:
            pickle.dump(data_tensor, file)

    # Run SVD on frontal slices and save U, S, and V for later
