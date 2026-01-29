%% Preview RGB Images from NTIRE2020_realworld Dataset
% This script generates RGB previews for all .mat files in the specified directory
% to help you select which image to use

clear; clc;
close all;

%% Configuration
% Get script directory for relative paths
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

% Data directory
data_dir = fullfile(script_dir, '..', '..', '..', 'exp', 'NTIRE2020_realworld', 'edsr');

% Wavelength configuration for NTIRE2020_realworld (31 channels)
lam31 = [400 410 420 430 440 450 460 470 480 490 500 510 ...
    520 530 540 550 560 570 580 590 600 610 620 630 ...
    640 650 660 670 680 690 700];

% RGB channels for NTIRE2020_realworld: R=650nm (ch26), G=550nm (ch16), B=450nm (ch6)
rgb_channels = [26, 16, 6];

fprintf('Scanning directory: %s\n', data_dir);

%% Get all .mat files
mat_files = dir(fullfile(data_dir, '*.mat'));
if isempty(mat_files)
    error('No .mat files found in: %s', data_dir);
end

fprintf('Found %d .mat files\n', length(mat_files));

%% Load and process each file
valid_files = {};
rgb_images = {};
file_names = {};

for i = 1:length(mat_files)
    file_name = mat_files(i).name;
    file_path = fullfile(data_dir, file_name);
    
    fprintf('\n[%d/%d] Processing: %s\n', i, length(mat_files), file_name);
    
    % Check file size
    file_info = dir(file_path);
    if isempty(file_info) || file_info.bytes == 0
        fprintf('  ⚠️  Skipping: File is empty or corrupted (size: %d bytes)\n', ...
            file_info.bytes);
        continue;
    end
    
    try
        % Load data
        data_struct = load(file_path);
        
        % Extract HSI data
        if isfield(data_struct, 'cube')
            hsi_data = data_struct.cube;
        elseif isfield(data_struct, 'pred')
            pred_data = data_struct.pred;
            if ndims(pred_data) == 4
                hsi_data = squeeze(pred_data(1, :, :, :));
            else
                hsi_data = pred_data;
            end
            hsi_data = flip(flip(hsi_data, 1), 2);
        else
            % Try to find any 3D array
            field_names = fieldnames(data_struct);
            found = false;
            for j = 1:length(field_names)
                var_name = field_names{j};
                if ~strcmp(var_name(1), '_')
                    var_data = data_struct.(var_name);
                    if ndims(var_data) == 3 || ndims(var_data) == 4
                        hsi_data = var_data;
                        if ndims(var_data) == 4
                            hsi_data = squeeze(hsi_data(1, :, :, :));
                        end
                        found = true;
                        break;
                    end
                end
            end
            if ~found
                fprintf('  ⚠️  Skipping: No 3D data found\n');
                continue;
            end
        end
        
        [h, w, num_channels] = size(hsi_data);
        fprintf('  ✓ Loaded: %d x %d x %d channels\n', h, w, num_channels);
        
        % Check if we have enough channels
        if num_channels < max(rgb_channels)
            fprintf('  ⚠️  Skipping: Not enough channels (%d < %d)\n', ...
                num_channels, max(rgb_channels));
            continue;
        end
        
        % Generate RGB image
        rgb_from_hsi = zeros(h, w, 3);
        for ch_idx = 1:3
            ch = rgb_channels(ch_idx);
            rgb_from_hsi(:, :, ch_idx) = hsi_data(:, :, ch);
        end
        
        % Normalize to [0, 1] range
        rgb_from_hsi = double(rgb_from_hsi);
        for ch_idx = 1:3
            ch_data = rgb_from_hsi(:, :, ch_idx);
            ch_data = ch_data - min(ch_data(:));
            if max(ch_data(:)) > 0
                ch_data = ch_data / max(ch_data(:));
            end
            rgb_from_hsi(:, :, ch_idx) = ch_data;
        end
        
        % Store results
        valid_files{end+1} = file_path;
        rgb_images{end+1} = rgb_from_hsi;
        file_names{end+1} = file_name;
        
        fprintf('  ✓ RGB generated successfully\n');
        
    catch ME
        fprintf('  ✗ Error: %s\n', ME.message);
        continue;
    end
end

%% Display all RGB images in a grid
fprintf('\n=== Displaying %d RGB previews ===\n', length(rgb_images));

if isempty(rgb_images)
    error('No valid files found to display');
end

% Calculate grid layout
num_images = length(rgb_images);
cols = ceil(sqrt(num_images));
rows = ceil(num_images / cols);

% Create figure
fig = figure('Name', 'RGB Previews - NTIRE2020_realworld', ...
    'NumberTitle', 'off', ...
    'Position', [100, 100, cols*300, rows*300]);

for i = 1:num_images
    subplot(rows, cols, i);
    imshow(rgb_images{i});
    title(file_names{i}, 'Interpreter', 'none', 'FontSize', 10);
end

fprintf('\n=== Preview Complete ===\n');
fprintf('Valid files found: %d\n', length(valid_files));
fprintf('File names:\n');
for i = 1:length(file_names)
    fprintf('  %d. %s\n', i, file_names{i});
end
fprintf('\nYou can now select which file to use based on the previews above.\n');