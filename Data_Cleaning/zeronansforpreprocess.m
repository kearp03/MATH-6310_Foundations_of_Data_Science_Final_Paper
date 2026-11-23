clear;
clc;

% ===============================
%% STEP 1: Define file locations
% ===============================

% Folder containing original NetCDF files
dataFolder = ...
'/Users/davidalejandrocruz/Downloads/SMAP-HB_surface-soil-moisture_30m_daily_netcdf';

% Path to "accepted-days" CSV.
% Move the attached CSV into dataFolder first
acceptedCSV = ...
'/Users/davidalejandrocruz/Downloads/SM_Accepted_Days_Under_10_Percent_NaN.csv';

% Folder for the prepared (month) (year) dataset
outputFolder = fullfile(dataFolder, ...
    'Prepared_June_2015');

% Create output folder if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% =========================================
%% STEP 2: Read accepted observation dates
% =========================================

% QC = Quality Control
QC = readtable(acceptedCSV);

% Select accepted observations from (month) (year)
selected = QC(QC.Year == 2015 & ...
              QC.Month == 6 & ...
              QC.NaNFraction < 0.10, :);

% Sort observations by day index
selected = sortrows(selected, 'DayIndex');

% Confirm the expected number of accepted days
assert(height(selected) == 17, ...
    'Expected 17 accepted June 2015 observations.');

fprintf('Preparing %d daily matrices.\n', height(selected));

% ===============================================
%% STEP 3: Read and preprocess each daily matrix
% ===============================================

for t = 1:height(selected)

    % Extract the source NetCDF filename
    sourceName = string(selected.SourceFile(t));

    % Construct its full path
    filePath = fullfile(dataFolder, sourceName);

    % Extract the day index from the CSV
    dayIndex = selected.DayIndex(t);

    % Read only the required daily matrix
    D = ncread(filePath, 'SMAPHB_SM', ...
        [1 1 dayIndex], [Inf Inf 1]);

    % Identify original missing-value locations
    missingMask = isnan(D);

    % Replace NaNs with zero for SVD/HOSVD
    D(missingMask) = 0;

    % Preserve memory-efficient single precision
    D = single(D);

    % Verify all entries are finite
    assert(all(isfinite(D(:))), ...
        'Nonfinite entries remain in the matrix.');

    % ====================================
    %% STEP 4: Save the processed matrix
    % ====================================

    % Create filename
    outputName = sprintf( ...
        'SM_2015_06_%02d.mat', dayIndex);

    % Save using compressed MATLAB v7 format
    save(fullfile(outputFolder, outputName), ...
        'D', '-v7');
    
    % =======================================
    %% STEP 5: Verify the missing-value mask
    % =======================================

    if t == 1

        % Preserve the reference mask
        referenceMask = missingMask;

        % Save the mask once
        save(fullfile(outputFolder, ...
            'missingMask.mat'), ...
            'referenceMask', '-v7');

    else

        % Confirm the mask matches the first day
        assert(isequal(missingMask, referenceMask), ...
            'Missing-value mask differs on day %d.', ...
            dayIndex);

    end

    % Display progress
    fprintf('Saved %s (%d of %d)\n', ...
        outputName, t, height(selected));

    % Clear temporary daily arrays
    clear D missingMask

end

% =====================================
%% STEP 6: Export the exact dates used
% =====================================

% Save the selected observations as a manifest
writetable(selected, ...
    fullfile(outputFolder, ...
    'June_2015_Manifest.csv'));

fprintf('\nJune 2015 preprocessing complete.\n');