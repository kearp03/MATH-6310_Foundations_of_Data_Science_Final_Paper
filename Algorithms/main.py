import numpy as np
import scipy.io
from pathlib import Path
import SVD
from helper_functions import *
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

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
            april_save_directory = parent_directory / 'April_Files'
            may_save_directory = parent_directory / 'May_Files'
            june_save_directory = parent_directory / 'June_Files'
            data_directory = parent_directory / 'Prepared_All_2015'
            save_directory = parent_directory / 'All_Months_Files'
            try:
                U_matrices = []
                S_vectors = []
                V_matrices = []
                U_matrices += read_pickle(april_save_directory / 'U_matrices.pkl')
                U_matrices += read_pickle(may_save_directory / 'U_matrices.pkl')
                U_matrices += read_pickle(june_save_directory / 'U_matrices.pkl')
                S_vectors += read_pickle(april_save_directory / 'S_vectors.pkl')
                S_vectors += read_pickle(may_save_directory / 'S_vectors.pkl')
                S_vectors += read_pickle(june_save_directory / 'S_vectors.pkl')
                V_matrices += read_pickle(april_save_directory / 'V_matrices.pkl')
                V_matrices += read_pickle(may_save_directory / 'V_matrices.pkl')
                V_matrices += read_pickle(june_save_directory / 'V_matrices.pkl')
                save_pickle(U_matrices, save_directory / 'U_matrices.pkl')
                save_pickle(S_vectors, save_directory / 'S_vectors.pkl')
                save_pickle(V_matrices, save_directory / 'V_matrices.pkl')
                print('Previous SVD files loaded successfully')
            except FileNotFoundError:
                print('Please run SVD on each month first to load previous SVD files')
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
    try:
        U_matrices = read_pickle(save_directory / 'U_matrices.pkl')
        S_vectors = read_pickle(save_directory / 'S_vectors.pkl')
        V_matrices = read_pickle(save_directory / 'V_matrices.pkl')
        print('SVD matrix components read successfully')
    except FileNotFoundError:
        print('Computing SVD of daily frontal slices')
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
        save_pickle(U_matrices, save_directory / 'U_matrices.pkl')
        save_pickle(S_vectors, save_directory / 'S_vectors.pkl')
        save_pickle(V_matrices, save_directory / 'V_matrices.pkl')

    # Test different reconstructed tensors using different truncated SVDs
    fro_errors: list[float] = []
    elem_stored: list[int] = []
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
    try:
        U1 = read_pickle(save_directory / 'U1_HOSVD_Householder.pkl')
        B1 = read_pickle(save_directory / 'B1_HOSVD_Householder.pkl')
        print('Householder Reduction read successfully')
    except FileNotFoundError:
        print('Running Householder Reduction')
        U1, B1 = SVD.householderReductionHO(flattened_tensor)
        save_pickle(U1, save_directory / 'U1_HOSVD_Householder.pkl')
        save_pickle(B1, save_directory / 'B1_HOSVD_Householder.pkl')
    try:
        U1 = read_pickle(save_directory / 'U1_HOSVD.pkl')
        # B1 = read_pickle(save_directory / 'B1_HOSVD.pkl')
        print('Left Singular Values read successfully')
    except FileNotFoundError:
        print('Running Golub Reinsch Iterations')
        U1, B1 = SVD.golubReinschHOSVD(U1, B1, 10000, threshold)
        save_pickle(U1, save_directory / 'U1_HOSVD.pkl')
        # save_pickle(B1, save_directory / 'B1_HOSVD.pkl')

    print('Running second SVD')
    flattened_tensor = tensor_to_matrix(data_tensor, 1)
    try:
        U2 = read_pickle(save_directory / 'U2_HOSVD_Householder.pkl')
        B2 = read_pickle(save_directory / 'B2_HOSVD_Householder.pkl')
        print('Householder Reduction read successfully')
    except FileNotFoundError:
        print('Running Householder Reduction')
        U2, B2 = SVD.householderReductionHO(flattened_tensor)
        save_pickle(U2, save_directory / 'U2_HOSVD_Householder.pkl')
        save_pickle(B2, save_directory / 'B2_HOSVD_Householder.pkl')
    try:
        U2 = read_pickle(save_directory / 'U2_HOSVD.pkl')
        # B2 = read_pickle(save_directory / 'B2_HOSVD.pkl')
        print('Left Singular Values read successfully')
    except FileNotFoundError:
        print('Running Golub Reinsch Iterations')
        U2, B2 = SVD.golubReinschHOSVD(U2, B2, 10000, threshold)
        save_pickle(U2, save_directory / 'U2_HOSVD.pkl')
        # save_pickle(B2, save_directory / 'B2_HOSVD.pkl')

    print('Running third SVD')
    flattened_tensor = tensor_to_matrix(data_tensor, 2)
    try:
        U3 = read_pickle(save_directory / 'U3_HOSVD_Householder.pkl')
        B3 = read_pickle(save_directory / 'B3_HOSVD_Householder.pkl')
        print('Householder Reduction read successfully')
    except FileNotFoundError:
        print('Running Householder Reduction')
        U3, B3 = SVD.householderReductionHO(flattened_tensor)
        save_pickle(U3, save_directory / 'U3_HOSVD_Householder.pkl')
        save_pickle(B3, save_directory / 'B3_HOSVD_Householder.pkl')
    try:
        U3 = read_pickle(save_directory / 'U3_HOSVD.pkl')
        # B3 = read_pickle(save_directory / 'B3_HOSVD.pkl')
        print('Left Singular Values read successfully')
    except FileNotFoundError:
        print('Running Golub Reinsch Iterations')
        U3, B3 = SVD.golubReinschHOSVD(U3, B3, 10000, threshold)
        save_pickle(U3, save_directory / 'U3_HOSVD.pkl')
        # save_pickle(B1, save_directory / 'B3_HOSVD.pkl')

    # Test different reconstructed tensors using different truncated HOSVDs
    fro_errors = []
    elem_stored = []
    for rank in ranks:
        print(f'Rank = {rank}')
        Sr = tensor_reconstruction(data_tensor, U1[:, :rank].T, U2[:, :rank].T, U3[:, :min(rank, U3.shape[1])].T)
        approx_data_tensor = tensor_reconstruction(Sr, U1[:, :rank], U2[:, :rank], U3[:, :min(rank, U3.shape[1])])
        fro_errors.append(frobenius_norm(data_tensor - approx_data_tensor))
        elem_stored.append(Sr.size + U1[:, :rank].size + U2[:, :rank].size + U3[:, :min(rank, U3.shape[1])].size)

    # Save results as pickle files
    save_pickle(ranks, save_directory / 'ranks.pkl')
    save_pickle(fro_errors, save_directory / 'fro_errors_HOSVD.pkl')
    save_pickle(elem_stored, save_directory / 'elem_stored_HOSVD.pkl')

def generate_figure(data_tensor: np.ndarray, save_directory: Path):
    # Pull in data from pickle files
    ranks = np.array(read_pickle(save_directory / 'ranks.pkl'))
    errors_SVD = np.array(read_pickle(save_directory / 'fro_errors_SVD.pkl'))
    elem_stored_SVD = np.array(read_pickle(save_directory / 'elem_stored_SVD.pkl')) / data_tensor.size
    errors_HOSVD = np.array(read_pickle(save_directory / 'fro_errors_HOSVD.pkl'))
    elem_stored_HOSVD = np.array(read_pickle(save_directory / 'elem_stored_HOSVD.pkl')) / data_tensor.size

    # Generate rank vs. error plot
    plt.plot(ranks, errors_SVD, label='Frontal Slice SVD')
    plt.plot(ranks, errors_HOSVD, label='HOSVD')
    plt.title('Rank of Truncated SVD and HOSVD vs. Frobenius norm error')
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(save_directory / 'Rank_vs_Error.eps', format='eps', bbox_inches='tight')
    plt.clf()

    # Generate rel. elements stored vs. error
    plt.plot(elem_stored_SVD, errors_SVD, label='Frontal Slice SVD')
    plt.plot(elem_stored_HOSVD, errors_HOSVD, label='HOSVD')
    plt.title('Relative Number of Elements Stored vs. Error')
    plt.xlabel('Relative Number of Elements Stored')
    plt.ylabel('Error')
    plt.savefig(save_directory / 'Elem_Stored_vs_Error.eps', format='eps', bbox_inches='tight')
    plt.clf()

def generate_heatmaps(save_directory: Path, slice: int, ranks: list[int]):
    data_tensor = read_pickle(save_directory / 'data_tensor.pkl')
    U_matrices = read_pickle(save_directory / 'U_matrices.pkl')
    S_vectors = read_pickle(save_directory / 'S_vectors.pkl')
    V_matrices = read_pickle(save_directory / 'V_matrices.pkl')
    U1 = read_pickle(save_directory / 'U1_HOSVD.pkl')
    U2 = read_pickle(save_directory / 'U2_HOSVD.pkl')
    U3 = read_pickle(save_directory / 'U3_HOSVD.pkl')

    ax = sns.heatmap(np.rot90(data_tensor[:,:,slice]), cmap='viridis', annot=False)
    ax.set(title=f'Original Slice {slice}')
    ax.tick_params(left=False, bottom=False)
    plt.savefig(save_directory / 'base_heatmap.eps', format='eps', bbox_inches='tight')
    plt.clf()

    for rank in ranks:
        approx_data_tensor = matrix_reconstruction(S_vectors[slice], U_matrices[slice], V_matrices[slice], rank)
        ax = sns.heatmap(np.rot90(approx_data_tensor), cmap='viridis', annot=False)
        ax.set(title=f'{rank} Rank Approximation SVD; Slice {slice}')
        ax.tick_params(left=False, bottom=False)
        plt.savefig(save_directory / f'{rank}_SVD_heatmap_{slice}.eps', format='eps', bbox_inches='tight')
        plt.clf()

        Sr = tensor_reconstruction(data_tensor, U1[:, :rank].T, U2[:, :rank].T, U3[:, :min(rank, U3.shape[1])].T)
        approx_data_tensor = tensor_reconstruction(Sr, U1[:, :rank], U2[:, :rank], U3[:, :min(rank, U3.shape[1])])
        sns.heatmap(np.rot90(approx_data_tensor[:, :, slice]), cmap='viridis', annot=False)
        ax.set(title=f'{rank} Rank Approximation HOSVD; Slice {slice}')
        ax.tick_params(left=False, bottom=False)
        plt.savefig(save_directory / f'{rank}_HOSVD_heatmap_{slice}.eps', format='eps', bbox_inches='tight')
        plt.clf()

def generate_comparison_figures(save_directory: Path):
    # Get each dataset's data tensor and save directories
    april_data_tensor, april_save_directory = get_data('april')
    may_data_tensor, may_save_directory = get_data('may')
    june_data_tensor, june_save_directory = get_data('june')
    all_data_tensor, all_save_directory = get_data('all')

    # Read in all data
    april_ranks = np.array(read_pickle(april_save_directory / 'ranks.pkl'))
    april_errors_SVD = np.array(read_pickle(april_save_directory / 'fro_errors_SVD.pkl')) / frobenius_norm(april_data_tensor)
    april_elem_stored_SVD = np.array(read_pickle(april_save_directory / 'elem_stored_SVD.pkl')) / april_data_tensor.size
    april_errors_HOSVD = np.array(read_pickle(april_save_directory / 'fro_errors_HOSVD.pkl')) / frobenius_norm(april_data_tensor)
    april_elem_stored_HOSVD = np.array(read_pickle(april_save_directory / 'elem_stored_HOSVD.pkl')) / april_data_tensor.size

    may_ranks = np.array(read_pickle(may_save_directory / 'ranks.pkl'))
    may_errors_SVD = np.array(read_pickle(may_save_directory / 'fro_errors_SVD.pkl')) / frobenius_norm(may_data_tensor)
    may_elem_stored_SVD = np.array(read_pickle(may_save_directory / 'elem_stored_SVD.pkl')) / may_data_tensor.size
    may_errors_HOSVD = np.array(read_pickle(may_save_directory / 'fro_errors_HOSVD.pkl')) / frobenius_norm(may_data_tensor)
    may_elem_stored_HOSVD = np.array(read_pickle(may_save_directory / 'elem_stored_HOSVD.pkl')) / may_data_tensor.size

    june_ranks = np.array(read_pickle(june_save_directory / 'ranks.pkl'))
    june_errors_SVD = np.array(read_pickle(june_save_directory / 'fro_errors_SVD.pkl')) / frobenius_norm(june_data_tensor)
    june_elem_stored_SVD = np.array(read_pickle(june_save_directory / 'elem_stored_SVD.pkl')) / june_data_tensor.size
    june_errors_HOSVD = np.array(read_pickle(june_save_directory / 'fro_errors_HOSVD.pkl')) / frobenius_norm(june_data_tensor)
    june_elem_stored_HOSVD = np.array(read_pickle(june_save_directory / 'elem_stored_HOSVD.pkl')) / june_data_tensor.size

    all_ranks = np.array(read_pickle(all_save_directory / 'ranks.pkl'))
    all_errors_SVD = np.array(read_pickle(all_save_directory / 'fro_errors_SVD.pkl')) / frobenius_norm(all_data_tensor)
    all_elem_stored_SVD = np.array(read_pickle(all_save_directory / 'elem_stored_SVD.pkl')) / all_data_tensor.size
    all_errors_HOSVD = np.array(read_pickle(all_save_directory / 'fro_errors_HOSVD.pkl')) / frobenius_norm(all_data_tensor)
    all_elem_stored_HOSVD = np.array(read_pickle(all_save_directory / 'elem_stored_HOSVD.pkl')) / all_data_tensor.size

    # Compare Rank vs. Rel. Error SVD
    plt.plot(april_ranks, april_errors_SVD, label='April')
    plt.plot(may_ranks, may_errors_SVD, label='May')
    plt.plot(june_ranks, june_errors_SVD, label='June')
    plt.plot(all_ranks, all_errors_SVD, label='All Months')
    plt.title('Rank of Truncated SVD vs. Relative Frobenius norm error by Month')
    plt.xlabel('Rank')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.savefig(save_directory / 'Rank_vs_Error_by_Month_SVD.png', format='png', bbox_inches='tight')
    plt.clf()

    # Compare Elements Stored vs. Rel. Error SVD
    plt.plot(april_elem_stored_SVD, april_errors_SVD, label='April')
    plt.plot(may_elem_stored_SVD, may_errors_SVD, label='May')
    plt.plot(june_elem_stored_SVD, april_errors_SVD, label='June')
    plt.plot(all_elem_stored_SVD, april_errors_SVD, label='All Months')
    plt.title('Relative Number of Elements Stored in SVD vs. Relative Error by Month')
    plt.xlabel('Relative Number of Elements Stored')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.savefig(save_directory / 'Elem_Stored_vs_Error_by_Month_SVD.png', format='png', bbox_inches='tight')
    plt.clf()

    # Compare Rank vs. Rel. Error HOSVD
    plt.plot(april_ranks, april_errors_HOSVD, label='April')
    plt.plot(may_ranks, may_errors_HOSVD, label='May')
    plt.plot(june_ranks, june_errors_HOSVD, label='June')
    plt.plot(all_ranks, all_errors_HOSVD, label='All Months')
    plt.title('Rank of Truncated HOSVD vs. Relative Frobenius norm error by Month')
    plt.xlabel('Rank')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.savefig(save_directory / 'Rank_vs_Error_by_Month_HOSVD.png', format='png', bbox_inches='tight')
    plt.clf()

    # Compare Elements Stored vs. Rel. Error HOSVD
    plt.plot(april_elem_stored_HOSVD, april_errors_HOSVD, label='April')
    plt.plot(may_elem_stored_HOSVD, may_errors_HOSVD, label='May')
    plt.plot(june_elem_stored_HOSVD, april_errors_HOSVD, label='June')
    plt.plot(all_elem_stored_HOSVD, april_errors_HOSVD, label='All Months')
    plt.title('Relative Number of Elements Stored in HOSVD vs. Relative Error by Month')
    plt.xlabel('Relative Number of Elements Stored')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.savefig(save_directory / 'Elem_Stored_vs_Error_by_Month_HOSVD.png', format='png', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    # Threshold for SVD algorithms
    threshold = 1e-8
    
    # Determine which month to analyze
    month = input('Enter month to analyze: ').lower()
    data_tensor, save_directory = get_data(month)

    s = timeit.default_timer()
    # Ranks to test reconstructions at
    ranks: list[int] = [1, 5, 10, 25, 30, 50, 100, 500, 1000, 2000, 3000, 3276]

    # Run SVD on frontal slices and save results to pickle files
    run_SVD(data_tensor, save_directory, ranks, threshold)

    # Run HOSVD on whole tensor and save results to pickle files
    run_HOSVD(data_tensor, save_directory, ranks, threshold)

    generate_figure(data_tensor=data_tensor, save_directory=save_directory)

    monthly_comp_save_directory = Path(__file__).resolve().parent.parent / 'Month_Comparisons'
    monthly_comp_save_directory.mkdir(parents=True, exist_ok=True)
    generate_comparison_figures(monthly_comp_save_directory)

    generate_heatmaps(save_directory, 1, [1, 500, 3000])

    e = timeit.default_timer()

    print(f"Elapsed: {round((e - s) * 1e6, 3)} µs")