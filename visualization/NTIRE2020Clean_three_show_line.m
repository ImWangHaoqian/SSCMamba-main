%% Plot Spectral Curves Comparison for ARAD_HS_0456 (NTIRE2020_clean)
% This script generates spectral curves comparison for 14 methods (including GT)
% Similar to show_line.m but adapted for NTIRE2020_clean dataset

clear; clc;
close all;

%% Configuration
% Get script directory for relative paths
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

% Method names (including GT - Ground Truth)
methods = {'hscnn_plus', 'hrnet', 'edsr', 'hdnet', 'mirnet', 'restormer', ...
    'hinet', 'mprnet', 'mst', 'mst_plus_plus', 'ssthyper', ...
    'gmsr', 'ssmamba', 'GT'};

% Data file path template (relative to script location)
data_base_path = fullfile(script_dir, '..', '..', '..', 'exp', 'NTIRE2020_clean');
data_file_name = 'ARAD_HS_0451.mat';

% Wavelength configuration for NTIRE2020_clean (31 channels)
% Based on show_simulation.m, NTIRE2020 uses 31 channels (400-700nm, 10nm steps)
lam31 = [400 410 420 430 440 450 460 470 480 490 500 510 ...
    520 530 540 550 560 570 580 590 600 610 620 630 ...
    640 650 660 670 680 690 700];

% Output directory (relative to script location)
output_base_dir = fullfile(script_dir, 'comparison_results', 'NTIRE2020_clean_ARAD_HS_0451');
if ~exist(output_base_dir, 'dir')
    mkdir(output_base_dir);
end

%% Step 1: Load GT Data and Select Region of Interest
fprintf('Step 1: Loading GT data for region selection...\n');

% Load GT data
gt_data_file_path = fullfile(data_base_path, 'GT', data_file_name);
if ~exist(gt_data_file_path, 'file')
    error('GT file not found: %s\nPlease check the data path.', gt_data_file_path);
end

fprintf('Loading GT data from: %s\n', gt_data_file_path);
gt_hsi_data = load_hsi_data(gt_data_file_path);
[h, w, num_channels] = size(gt_hsi_data);
fprintf('GT data size: %d x %d x %d channels\n', h, w, num_channels);

% Generate RGB image from hyperspectral data for region selection
% Use channels that correspond to Red, Green, Blue wavelengths
% For NTIRE2020_clean (31 channels, 400-700nm, 10nm steps):
% - Channel 6 (450nm) - Blue region
% - Channel 16 (550nm) - Green region  
% - Channel 26 (650nm) - Red region
if num_channels >= 26
    rgb_channels = [26, 16, 6];  % [R, G, B] channel indices for NTIRE2020_clean
else
    % Fallback: use first, middle, last channels
    rgb_channels = [num_channels, round(num_channels/2), 1];
end

% Extract RGB channels and normalize
rgb_from_hsi = zeros(h, w, 3);
for i = 1:3
    ch = rgb_channels(i);
    if ch <= num_channels
        rgb_from_hsi(:, :, i) = gt_hsi_data(:, :, ch);
    end
end

% Normalize to [0, 1] range
rgb_from_hsi = double(rgb_from_hsi);
for i = 1:3
    ch_data = rgb_from_hsi(:, :, i);
    ch_data = ch_data - min(ch_data(:));
    if max(ch_data(:)) > 0
        ch_data = ch_data / max(ch_data(:));
    end
    rgb_from_hsi(:, :, i) = ch_data;
end

% Let user manually select region using getrect
fprintf('\nPlease select the region of interest in the image window...\n');
fprintf('Instructions:\n');
fprintf('  1. Click and drag to create a rectangle\n');
fprintf('  2. Release mouse button to finish selection\n');
fprintf('This region will be used for all methods to calculate spectral curves.\n\n');

figure('Name', 'Select Region of Interest', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
imshow(rgb_from_hsi);
title('GT RGB - Click and drag to select region for spectral analysis');

% Use getrect which returns [x_min, y_min, width, height]
rect_crop = getrect(gca);

% Check if selection was cancelled or invalid
if isempty(rect_crop) || length(rect_crop) ~= 4
    close;
    error('Region selection was cancelled. Please run the script again and select a valid region.');
end

% Check if width and height are valid
if rect_crop(3) <= 0 || rect_crop(4) <= 0
    close;
    error('Invalid region selected (width=%.1f, height=%.1f).\nPlease drag a rectangle with positive width and height.', ...
        rect_crop(3), rect_crop(4));
end

% Round to integers
rect_crop = round(rect_crop);
close;

fprintf('Selected crop region: x=%d, y=%d, width=%d, height=%d\n', ...
    rect_crop(1), rect_crop(2), rect_crop(3), rect_crop(4));

% Validate crop region
if rect_crop(3) < 10 || rect_crop(4) < 10
    error('Selected region is too small (width=%d, height=%d). Please select a larger region (at least 10x10 pixels).', ...
        rect_crop(3), rect_crop(4));
end

%% Step 2: Load Data and Calculate Mean Spectral Curves
fprintf('\nStep 2: Loading data and calculating mean spectral curves...\n');

% Determine actual number of channels and adjust wavelength vector
% For NTIRE2020_clean, data should have 31 channels matching lam31
actual_num_channels = num_channels;
if actual_num_channels == length(lam31)
    channels_to_use = 1:actual_num_channels;
    wavelengths_to_use = lam31;
else
    % Create wavelength vector based on actual channels
    fprintf('Data has %d channels, creating wavelength vector...\n', actual_num_channels);
    % Estimate wavelengths: 400nm to 700nm range (NTIRE2020_clean)
    wavelengths_to_use = linspace(400, 700, actual_num_channels);
    channels_to_use = 1:actual_num_channels;
end

% Initialize storage for spectral curves
spec_mean_all = zeros(length(methods), length(channels_to_use));
corr_values = zeros(length(methods), 1);

% First, get GT spectral curve
fprintf('\nProcessing GT...\n');
x_start = max(1, round(rect_crop(1)));
y_start = max(1, round(rect_crop(2)));
x_end = min(w, x_start + round(rect_crop(3)) - 1);
y_end = min(h, y_start + round(rect_crop(4)) - 1);

gt_cropped = gt_hsi_data(y_start:y_end, x_start:x_end, channels_to_use);
spec_mean_gt = mean(mean(gt_cropped, 1), 2);
spec_mean_gt = spec_mean_gt(:);  % Convert to column vector
spec_mean_gt = spec_mean_gt ./ max(spec_mean_gt);  % Normalize
spec_mean_all(14, :) = spec_mean_gt';  % GT is last in methods list
corr_values(14) = 1.0;  % GT correlation with itself

fprintf('GT spectral curve calculated: %d channels\n', length(spec_mean_gt));

% Process each method
for method_idx = 1:length(methods)
    method_name = methods{method_idx};
    
    % Skip GT (already processed)
    if strcmp(method_name, 'GT')
        continue;
    end
    
    fprintf('\nProcessing method: %s (%d/%d)\n', method_name, method_idx, length(methods));
    
    % Load data file
    data_file_path = fullfile(data_base_path, method_name, data_file_name);
    
    if ~exist(data_file_path, 'file')
        fprintf('Warning: File not found: %s\n', data_file_path);
        fprintf('Skipping method: %s\n', method_name);
        continue;
    end
    
    try
        % Load data using helper function
        hsi_data = load_hsi_data(data_file_path);
        
        % Get data dimensions
        [h_method, w_method, num_channels_method] = size(hsi_data);
        fprintf('Loaded data: %d x %d x %d channels\n', h_method, w_method, num_channels_method);
        
        % Check if channels match (allow some flexibility)
        if num_channels_method < length(channels_to_use)
            fprintf('Warning: Method has fewer channels (%d) than required (%d), skipping...\n', ...
                num_channels_method, length(channels_to_use));
            continue;
        end
        
        % Crop region
        x_start = max(1, round(rect_crop(1)));
        y_start = max(1, round(rect_crop(2)));
        x_end = min(w_method, x_start + round(rect_crop(3)) - 1);
        y_end = min(h_method, y_start + round(rect_crop(4)) - 1);
        
        % Ensure valid indices
        if x_end < x_start || y_end < y_start
            fprintf('Warning: Invalid crop region for method %s, skipping...\n', method_name);
            continue;
        end
        
        % Crop data and use only the channels we need
        cropped_data = hsi_data(y_start:y_end, x_start:x_end, channels_to_use);
        fprintf('Cropped region: %d x %d x %d channels\n', ...
            size(cropped_data, 1), size(cropped_data, 2), size(cropped_data, 3));
        
        % Calculate mean spectral curve
        spec_mean = mean(mean(cropped_data, 1), 2);
        spec_mean = spec_mean(:);  % Convert to column vector
        
        % Ensure correct length
        if length(spec_mean) ~= length(channels_to_use)
            fprintf('Warning: Spectral curve length mismatch, skipping...\n');
            continue;
        end
        
        % Normalize
        if max(spec_mean) > 0
            spec_mean = spec_mean ./ max(spec_mean);
        end
        
        % Store spectral curve
        spec_mean_all(method_idx, :) = spec_mean';
        
        % Calculate correlation with GT
        corr_value = corr(spec_mean_gt(:), spec_mean(:));
        corr_values(method_idx) = corr_value;
        
        fprintf('  Spectral curve calculated, correlation with GT: %.4f\n', corr_value);
        
    catch ME
        fprintf('Error processing method %s: %s\n', method_name, ME.message);
        continue;
    end
end

%% Step 3: Save RGB Image with Red Box
fprintf('\nStep 3: Saving RGB image with red box...\n');

% Create RGB image with red bounding box
rgb_with_box = rgb_from_hsi;
% Draw red rectangle on the image
x1 = rect_crop(1);
y1 = rect_crop(2);
x2 = x1 + rect_crop(3) - 1;
y2 = y1 + rect_crop(4) - 1;

% Ensure coordinates are within image bounds
x1 = max(1, x1);
y1 = max(1, y1);
x2 = min(size(rgb_with_box, 2), x2);
y2 = min(size(rgb_with_box, 1), y2);

% Draw red rectangle (thick lines)
line_width = 3;
% Top and bottom lines
rgb_with_box(max(1, y1-line_width):y1, x1:x2, 1) = 1;  % Red channel
rgb_with_box(max(1, y1-line_width):y1, x1:x2, 2) = 0;  % Green channel
rgb_with_box(max(1, y1-line_width):y1, x1:x2, 3) = 0;  % Blue channel

rgb_with_box(y2:min(size(rgb_with_box, 1), y2+line_width), x1:x2, 1) = 1;
rgb_with_box(y2:min(size(rgb_with_box, 1), y2+line_width), x1:x2, 2) = 0;
rgb_with_box(y2:min(size(rgb_with_box, 1), y2+line_width), x1:x2, 3) = 0;

% Left and right lines
rgb_with_box(y1:y2, max(1, x1-line_width):x1, 1) = 1;
rgb_with_box(y1:y2, max(1, x1-line_width):x1, 2) = 0;
rgb_with_box(y1:y2, max(1, x1-line_width):x1, 3) = 0;

rgb_with_box(y1:y2, x2:min(size(rgb_with_box, 2), x2+line_width), 1) = 1;
rgb_with_box(y1:y2, x2:min(size(rgb_with_box, 2), x2+line_width), 2) = 0;
rgb_with_box(y1:y2, x2:min(size(rgb_with_box, 2), x2+line_width), 3) = 0;

% Save image with red box
rgb_box_filename = fullfile(output_base_dir, 'spectral_analysis_RGB_with_red_box.png');
imwrite(rgb_with_box, rgb_box_filename);
fprintf('Saved RGB image with red box: %s\n', rgb_box_filename);

%% Step 4: Plot Spectral Curves
fprintf('\nStep 4: Plotting spectral curves...\n');

% Prepare data for plotting
X = wavelengths_to_use;
Y = spec_mean_all;

% Remove rows with all zeros (methods that failed to load)
valid_rows = any(Y ~= 0, 2);
Y = Y(valid_rows, :);
corr_values_plot = corr_values(valid_rows);
methods_plot = methods(valid_rows);

fprintf('Plotting %d methods...\n', size(Y, 1));

% Create figure with curves
createfigure_14methods(X, Y, corr_values_plot, methods_plot);

%% Step 5: Save Figure
fprintf('\nStep 5: Saving figure...\n');

output_figure_path = fullfile(output_base_dir, 'spectral_comparison_ARAD_HS_0451.png');
saveas(gcf, output_figure_path);
fprintf('Saved figure: %s\n', output_figure_path);

fprintf('\n=== Spectral Curves Comparison Complete ===\n');

%% Helper Function: Load HSI Data from .mat file
function hsi_data = load_hsi_data(data_file_path)
    % Load .mat file and extract HSI data
    hsi_data = [];
    use_h5py = false;
    
    % Check file size first - if file is too small, it might be corrupted
    file_info = dir(data_file_path);
    if isempty(file_info)
        error('File does not exist: %s', data_file_path);
    end
    file_size = file_info.bytes;
    
    % Try multiple methods to load the file
    % Method 1: Standard load
    try
        data_struct = load(data_file_path);
        % Check if load succeeded but returned empty or invalid
        if isempty(fieldnames(data_struct))
            error('Loaded file is empty');
        end
    catch ME_load
        % Method 2: Try matfile (works with HDF5 v7.3 files)
        try
            m = matfile(data_file_path);
            if isprop(m, 'cube')
                hsi_data = m.cube;
                use_h5py = true;
            else
                % Try to find any 3D property
                props = properties(m);
                found = false;
                for i = 1:length(props)
                    try
                        var_data = m.(props{i});
                        if ndims(var_data) == 3 || ndims(var_data) == 4
                            hsi_data = var_data;
                            if ndims(var_data) == 4
                                hsi_data = squeeze(hsi_data(1, :, :, :));
                            end
                            found = true;
                            use_h5py = true;
                            break;
                        end
                    catch
                        continue;
                    end
                end
                if ~found
                    error('No 3D data found in matfile');
                end
            end
        catch ME_matfile
            % Method 3: Try MATLAB's low-level HDF5 functions directly
            try
                fprintf('  Attempting to read using MATLAB HDF5 functions...\n');
                
                % Check if file exists
                if ~exist(data_file_path, 'file')
                    error('File does not exist');
                end
                
                % Try to read using h5read
                if exist('h5read', 'file') == 2
                    try
                        % First, try to get file info
                        info = h5info(data_file_path);
                        
                        % Look for 'cube' dataset
                        dataset_path = '';
                        if isfield(info, 'Datasets')
                            for i = 1:length(info.Datasets)
                                if strcmp(info.Datasets(i).Name, 'cube')
                                    dataset_path = '/cube';
                                    break;
                                end
                            end
                        end
                        
                        % Also check Groups for nested datasets
                        if isempty(dataset_path) && isfield(info, 'Groups')
                            for i = 1:length(info.Groups)
                                group = info.Groups(i);
                                if isfield(group, 'Datasets')
                                    for j = 1:length(group.Datasets)
                                        if strcmp(group.Datasets(j).Name, 'cube')
                                            dataset_path = [group.Name '/cube'];
                                            break;
                                        end
                                    end
                                end
                                if ~isempty(dataset_path)
                                    break;
                                end
                            end
                        end
                        
                        % If still not found, try to find any 3D dataset
                        if isempty(dataset_path)
                            if isfield(info, 'Datasets')
                                for i = 1:length(info.Datasets)
                                    dataset_path = ['/' info.Datasets(i).Name];
                                    break;
                                end
                            end
                        end
                        
                        if ~isempty(dataset_path)
                            hsi_data = h5read(data_file_path, dataset_path);
                            % HDF5 data might be in different dimension order
                            if ndims(hsi_data) == 3
                                [d1, d2, d3] = size(hsi_data);
                                if d1 < 50 && d2 > 100 && d3 > 100
                                    hsi_data = permute(hsi_data, [2, 3, 1]);
                                end
                            end
                            use_h5py = true;
                            fprintf('  Successfully read using h5read from path: %s\n', dataset_path);
                        else
                            error('Could not find any dataset in HDF5 file');
                        end
                    catch ME_h5read
                        fprintf('  h5read failed: %s\n', ME_h5read.message);
                        error('h5read failed: %s', ME_h5read.message);
                    end
                else
                    error('HDF5 support (h5read) not available');
                end
            catch ME_h5
                error('Could not read file: %s\nStandard load: %s\nMatfile: %s\nH5read: %s', ...
                    data_file_path, ME_load.message, ME_matfile.message, ME_h5.message);
            end
        end
    end
    
    if ~use_h5py && isempty(hsi_data)
        % Detect variable name (cube or pred)
        if isfield(data_struct, 'cube')
            hsi_data = data_struct.cube;
        elseif isfield(data_struct, 'pred')
            pred_data = data_struct.pred;
            % Handle pred format
            if ndims(pred_data) == 4
                hsi_data = squeeze(pred_data(1, :, :, :));
            else
                hsi_data = pred_data;
            end
            % Apply flip if needed
            hsi_data = flip(flip(hsi_data, 1), 2);
        else
            % Try to find any 3D array
            field_names = fieldnames(data_struct);
            found = false;
            for i = 1:length(field_names)
                var_name = field_names{i};
                % Skip MATLAB metadata fields
                if strcmp(var_name(1), '_')
                    continue;
                end
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
            if ~found
                error('Could not find HSI data in file: %s', data_file_path);
            end
        end
    end
    
    % Ensure data is in correct format: [height, width, channels]
    if size(hsi_data, 3) == 1 && (size(hsi_data, 1) == 1 || size(hsi_data, 2) == 1)
        error('Data format error: expected 3D array [H, W, C] in file: %s', data_file_path);
    end
end

%% Helper Function: Create Figure with 14 Methods
function createfigure_14methods(X1, YMatrix1, Corr, method_names)
    %CREATEFIGURE_14METHODS(X1, YMatrix1, Corr, method_names)
    %  X1:  x 数据的向量 (wavelengths)
    %  YMatrix1:  y 数据的矩阵 (14 rows x 28 columns)
    %  Corr: 相关系数向量 (14 elements)
    %  method_names: 方法名称cell数组 (14 elements)
    
    % 创建 figure
    figure1 = figure('PaperOrientation','landscape',...
        'PaperSize',[29.69999902 20.99999864], 'Color', 'white');
    
    % 创建 axes
    axes1 = axes('Parent',figure1,'Position',[0.13 0.11 0.385625 0.815]);
    hold(axes1,'on');
    
    % 定义颜色方案（14种不同颜色）
    colors = [
        0.8  0.8  0.0;  % hscnn_plus - Yellow
        0.0  0.6  0.6;  % hrnet - Cyan
        0.0  0.5  1.0;  % edsr - Blue
        0.8  0.4  0.0;  % hdnet - Orange
        0.4  0.0  0.4;  % mirnet - Dark Purple
        0.8  0.0  0.4;  % restormer - Pink
        0.6  0.0  0.8;  % hinet - Purple
        0.8  0.6  0.4;  % mprnet - Brown
        0.0  0.4  0.8;  % mst - Dark Blue
        0.4  0.8  0.0;  % mst_plus_plus - Light Green
        0.6  0.4  0.2;  % ssthyper - Tan
        0.0  0.8  0.4;  % gmsr - Green
        0.2  0.6  0.8;  % ssmamba - Sky Blue
        1.0  0.0  0.0;  % GT - Red
    ];
    
    % 使用 plot 的矩阵输入创建多行
    plot1 = plot(X1, YMatrix1, 'MarkerSize', 12, 'Marker', '.', 'LineWidth', 2.0, ...
        'Parent', axes1);
    
    % 设置每条曲线的颜色和标签
    for i = 1:length(plot1)
        method_name = method_names{i};
        if strcmp(method_name, 'GT')
            display_name = ' Ground Truth';
        else
            display_name = sprintf(' %s, corr: %.4f', upper(method_name), Corr(i));
        end
        set(plot1(i), 'DisplayName', display_name, 'Color', colors(i, :));
    end
    
    % 设置Y轴范围
    ylim(axes1, [0 1]);
    box(axes1, 'on');
    hold(axes1, 'off');
    
    % 设置其余坐标区属性
    set(axes1, 'FontName', 'Arial', 'FontSize', 22, 'LineWidth', 3.5);
    
    % 创建 ylabel
    ylabel('Normalized Density', 'FontSize', 28, 'FontName', 'Arial');
    
    % 创建 xlabel
    xlabel('Wavelength (nm)', 'FontSize', 28, 'FontName', 'Arial');
    
    % 创建 legend
    legend1 = legend(axes1, 'show');
    set(legend1,...
        'Position', [0.55 0.1 0.35 0.8],...
        'FontSize', 16,...
        'EdgeColor', [1 1 1],...
        'NumColumns', 1);
end