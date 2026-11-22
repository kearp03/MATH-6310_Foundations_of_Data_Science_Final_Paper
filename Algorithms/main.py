import numpy as np
import scipy.io
from pathlib import Path
import SVD
from helper_functions import *
import timeit

def get_data(month: str):
    # Get directories as Paths
    current_directory = Path(__file__).resolve().parent
    parent_directory = current_directory.parent
    data_directory = Path()
    save_directory = Path()
    
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
        case 'all':
            data_directory = parent_directory / 'Prepared_All_2015'
            save_directory = parent_directory / 'All_Months_Files'
        case _:
            print('Month not available')
            exit()
    # Make directory for save files if it doesn't already exist
    save_directory.mkdir(parents=True, exist_ok=True)
    
    # Try to read in data tensor from pickle file
    try:
        data_tensor = read_pickle(save_directory / 'data_tensor.pkl')
        print('Data tensor read successfully')
    
    # If file hasn't been created yet, read in Matlab files to create it
    except FileNotFoundError:
        print('Creating data tensor')
        # Determine the number of days in the month from the number of files in data directory
        number_of_days = sum([1 for file in data_directory.glob('SM_*.mat') if file.is_file() and file.suffix.casefold() == '.mat'.casefold()])

        # Create an empty array to hold the soil moisture data
        data_tensor = np.zeros((3312,3276,number_of_days))

        # Load each day's data into each frontal slice of the data tensor
        for i, file_path in enumerate(sorted(data_directory.glob('SM_*.mat'))):
            data_tensor[:,:,i] = scipy.io.loadmat(file_path)['D']

        # Save the data tensor to be read easier later
        save_pickle(data_tensor, save_directory / 'data_tensor.pkl')
    return data_tensor, save_directory

def run_SVD(data_tensor: np.ndarray, save_directory: Path, ranks: list[int], threshold: float):
    # Run SVD on frontal slices and save U, S, and V for later
    U_matrices: list[np.ndarray] = []
    S_vectors: list[np.ndarray] = []
    V_matrices: list[np.ndarray] = []
    for i in range(data_tensor.shape[2]):
        print(i)
        daily_matrix = data_tensor[:,:,i].copy()

        U, B, V = SVD.householderReduction(daily_matrix)
        U, B, V = SVD.golubReinschSVD(U, B, V, iterations=10000, threshold=threshold)
        S = np.diag(B)
        U_matrices.append(U)
        S_vectors.append(S)
        V_matrices.append(V)

    # Test different reconstructed tensors using different truncated SVDs
    fro_errors = []
    elem_stored = []
    for i in range(len(ranks)):
        rank = ranks[i]
        approx_data_tensor = np.zeros_like(data_tensor)
        elem_stored.append(0)
        for j in range(data_tensor.shape[2]):
            approx_data_tensor[:,:,j] = matrix_reconstruction(S_vectors[j], U_matrices[j], V_matrices[j], rank)
            elem_stored[i] += S_vectors[j][:rank].size + U_matrices[j][:, :rank].size + V_matrices[j][:, :rank].size
        fro_errors.append(frobenius_norm(data_tensor - approx_data_tensor))
    
    # Save results as pickle files
    save_pickle(ranks, save_directory / 'ranks.pkl')
    save_pickle(fro_errors, save_directory / 'fro_errors_SVD.pkl')
    save_pickle(elem_stored, save_directory / 'elem_stored_SVD.pkl')

def run_HOSVD(data_tensor: np.ndarray, save_directory: Path, ranks: list[int], threshold: float):
    # Run HOSVD on entire data tensor
    print('Running first SVD')
    flattened_tensor = tensor_to_matrix(data_tensor, 0)
    U1, B1 = SVD.householderReductionHO(flattened_tensor)
    SVD.golubReinschHOSVD(U1, B1, 10000, threshold)

    print('Running second SVD')
    flattened_tensor = tensor_to_matrix(data_tensor, 1)
    U2, B2 = SVD.householderReductionHO(flattened_tensor)
    SVD.golubReinschHOSVD(U2, B2, 10000, threshold)

    print('Running third SVD')
    flattened_tensor = tensor_to_matrix(data_tensor, 2)
    U3, B3 = SVD.householderReductionHO(flattened_tensor)
    SVD.golubReinschHOSVD(U3, B3, 10000, threshold)

    # Test different reconstructed tensors using different truncated HOSVDs
    fro_errors = []
    elem_stored = []
    for rank in ranks:
        print(f'Rank = {rank}')
        Sr = tensor_reconstruction(data_tensor, U1[:, :rank].T, U2[:, :rank].T, U3[:, :min(rank, U3.shape[1])].T)
        approx_data_tensor = tensor_reconstruction(Sr, U1[:, :rank], U2[:, :rank], U3[:, :min(rank, U3.shape[1])])
        fro_errors.append(frobenius_norm(data_tensor - approx_data_tensor))
        elem_stored.append(Sr.shape + U1[:, :rank].shape + U2[:, :rank].shape + U3[:, :min(rank, U3.shape[1])].shape)

    # Save results as pickle files
    save_pickle(ranks, save_directory / 'ranks.pkl')
    save_pickle(fro_errors, save_directory / 'fro_errors_HOSVD.pkl')
    save_pickle(elem_stored, save_directory / 'elem_stored_HOSVD.pkl')

if __name__ == '__main__':
    # Threshold for SVD algorithms
    
    threshold = 1e-8
    
    # Determine which month to analyze
    month = input('Enter month to analyze: ').lower()
    data_tensor, save_directory = get_data(month)

    s = timeit.default_timer()
    # Ranks to test reconstructions at
    ranks = [1, 5, 10, 25, 30, 50, 100, 500, 1000, 2000, 3000, 3276]

    # Run SVD on frontal slices and save results to pickle files
    run_SVD(data_tensor, save_directory, ranks, threshold)

    # Run HOSVD on whole tensor and save results to pickle files
    run_HOSVD(data_tensor, save_directory, ranks, threshold)

    e = timeit.default_timer()

    print(f"Elapsed: {round((e - s) * 1e6, 3)} µs")