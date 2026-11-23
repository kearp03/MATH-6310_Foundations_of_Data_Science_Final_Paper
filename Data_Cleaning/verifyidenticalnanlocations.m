% Read the s.m. matrix starting from first longitude index, 
% first latitude index, on day 2 (year and month have been chosen in
% findgoodmatrices.m) Then read all values in the 1st mode, 2nd mode, for
% that day (hence, [Inf Inf 1])
D1 = ncread(filePath, 'SMAPHB_SM', ...
    [1 1 2], [Inf Inf 1]);

% Ditto, but for day 3
D2 = ncread(filePath, 'SMAPHB_SM', ...
    [1 1 3], [Inf Inf 1]);

% Create a logical matrices identifying NaN locations in day 2 and day 3.
mask1 = isnan(D1);
mask2 = isnan(D2);

% Compare both logical matrices to determine whether
% the NaN values occur at exactly the same grid locations.
sameMask = isequal(mask1, mask2);

% Display 1 if the NaN locations are identical, or 0 otherwise.
fprintf('Identical NaN locations: %d\n', sameMask);

% Open a new figure window for visualization.
figure;

% Display the missing-value mask for day 2 as an image.
% A value of 1 indicates NaN; a value of 0 indicates non-NaN.
imagesc(mask1);

% Set equal scaling along both spatial axes, add a color bar and title.
axis image;
colorbar;
title('Spatial Distribution of Missing Values');