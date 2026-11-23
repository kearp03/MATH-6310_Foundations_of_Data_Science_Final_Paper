
clear;
clc;

% User settings

year = 2015;
month = 4;

nanThreshold = 0.10;  % Reject if more than 10% NaN

dataFolder = ...
    '/Users/davidalejandrocruz/Downloads/SMAP-HB_surface-soil-moisture_30m_daily_netcdf';

% Construct filename
fileName = sprintf( ...
    'SMAP-HB_surface-soil-moisture_30m_daily_%04d-%02d.nc', ...
    year, month);

filePath = fullfile(dataFolder, fileName);

% Read metadata (not the full dataset)
info = ncinfo(filePath, 'SMAPHB_SM');

numDays = info.Size(3);

% Preallocate results
nanFraction = zeros(numDays, 1);
keepMatrix = false(numDays, 1);
dates = NaT(numDays, 1);

% Process one day at a time
for t = 1:numDays

    % Read only day t
    D = ncread(filePath, 'SMAPHB_SM', ...
        [1 1 t], [Inf Inf 1]);

    % Calculate proportion of NaN values
    nanFraction(t) = nnz(isnan(D)) / numel(D);

    % Determine whether to keep the matrix
    keepMatrix(t) = nanFraction(t) <= nanThreshold;

    % Record the corresponding date
    dates(t) = datetime(year, month, 1) + days(t-1);

    fprintf('Day %02d: %.2f%% NaN | Keep: %d\n', ...
        t, 100*nanFraction(t), keepMatrix(t));

    % Release daily matrix from memory
    clear D;

end


% ====================================================================
% All code below simply deals with creating master QC tables and PDFs
% for the good matrices that we accepted in the netCDF files
% ====================================================================

% Create summary table
QC = table(dates, nanFraction, keepMatrix, ...
    'VariableNames', {'Date', 'NaNFraction', 'Keep'});

disp(QC);

% Summary
fprintf('\nAccepted days: %d\n', sum(keepMatrix));
fprintf('Rejected days: %d\n', sum(~keepMatrix));

% Save results
outputName = sprintf('QC_%04d_%02d.csv', year, month);

writetable(QC, fullfile(dataFolder, outputName));


% Update master quality-control records
masterCSV = fullfile(dataFolder, 'SM_Master_QC.csv');
masterPDF = fullfile(dataFolder, 'SM_Master_QC.pdf');

% Add information useful for future processing
QC.Year = repmat(year, height(QC), 1);
QC.Month = repmat(month, height(QC), 1);
QC.DayIndex = (1:height(QC))';
QC.NaNPercent = 100 * QC.NaNFraction;
QC.Threshold = repmat(nanThreshold, height(QC), 1);
QC.SourceFile = repmat(string(fileName), height(QC), 1);

% Read previous results if the master file exists
if isfile(masterCSV)
    MasterQC = readtable(masterCSV);
    MasterQC.Date = datetime(MasterQC.Date);
else
    MasterQC = QC([],:);
end

% Remove old records for dates being processed again
MasterQC(ismember(MasterQC.Date, QC.Date), :) = [];

% Add the current month's results
MasterQC = [MasterQC; QC];

% Arrange all records chronologically
MasterQC = sortrows(MasterQC, 'Date');

% Save updated master CSV
writetable(MasterQC, masterCSV);

fprintf('\nMaster CSV updated: %s\n', masterCSV);
fprintf('Total daily records: %d\n', height(MasterQC));

%% Generate master PDF report

% Display 35 daily records per PDF page
rowsPerPage = 35;
numPages = ceil(height(MasterQC) / rowsPerPage);

for page = 1:numPages

    firstRow = (page - 1) * rowsPerPage + 1;
    lastRow = min(page * rowsPerPage, height(MasterQC));

    fig = figure('Visible', 'off', ...
        'Color', 'white', ...
        'Units', 'inches', ...
        'Position', [1 1 8.5 11]);

    ax = axes(fig, 'Position', [0.08 0.06 0.84 0.88]);
    axis(ax, [0 1 0 1]);
    axis(ax, 'off');

    % Report heading
    text(ax, 0, 0.98, 'SOIL MOISTURE QUALITY CONTROL', ...
        'FontSize', 15, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'top');

    subtitle = sprintf( ...
        'Master Report | %d Records | Page %d of %d', ...
        height(MasterQC), page, numPages);

    text(ax, 0, 0.945, subtitle, ...
        'FontSize', 10, 'VerticalAlignment', 'top');

    % Table header
    header = sprintf('%-12s  %8s  %9s  %7s', ...
        'Date', 'NaN (%)', 'Limit (%)', 'Keep');

    text(ax, 0, 0.90, header, ...
        'FontName', 'Courier', ...
        'FontSize', 10, 'FontWeight', 'bold');

    text(ax, 0, 0.88, repmat('-', 1, 49), ...
        'FontName', 'Courier', 'FontSize', 10);

    % Table rows
    for r = firstRow:lastRow

        position = r - firstRow;

        dateString = datestr(MasterQC.Date(r), 'yyyy-mm-dd');

        if MasterQC.Keep(r)
            status = 'YES';
        else
            status = 'NO';
        end

        line = sprintf('%-12s  %8.2f  %9.2f  %7s', ...
            dateString, ...
            MasterQC.NaNPercent(r), ...
            100 * MasterQC.Threshold(r), ...
            status);

        y = 0.85 - position * 0.022;

        text(ax, 0, y, line, ...
            'FontName', 'Courier', ...
            'FontSize', 9, ...
            'Interpreter', 'none');
    end

    % Export pages into one PDF
    if page == 1
        exportgraphics(fig, masterPDF, ...
            'ContentType', 'vector');
    else
        exportgraphics(fig, masterPDF, ...
            'ContentType', 'vector', ...
            'Append', true);
    end

    close(fig);
end

fprintf('Master PDF updated: %s\n', masterPDF);
