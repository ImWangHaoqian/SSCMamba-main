%% Generate Comparison Figures for ARAD_HS_0456 (NTIRE2020_clean)
% This script generates cropped region visualizations for 13 methods
% across 3 channels (26-Red, 16-Green, 6-Blue)

clear; clc;
close all;

%% Configuration
% Get script directory for relative paths
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

% Reference method for region selection (use first method by default)
reference_method_idx = 1;  % Will use this method's data to show RGB for region selection

% Output directory (relative to script location)
output_base_dir = fullfile(script_dir, 'comparison_results', 'NTIRE2020_clean_ARAD_HS_0451');
if ~exist(output_base_dir, 'dir')
    mkdir(output_base_dir);
end
fprintf('Output directory: %s\n', output_base_dir);
fprintf('Output directory exists: %d\n', exist(output_base_dir, 'dir'));

% Method names (including GT - Ground Truth)
methods = {'edsr', 'gmsr', 'hdnet', 'hinet', 'hrnet', 'hscnn_plus', ...
           'mirnet', 'mprnet', 'mst', 'mst_plus_plus', 'restormer', ...
           'ssmamba', 'ssthyper', 'GT'};

% Target channels for NTIRE2020_clean (31 channels, 400-700nm, 10nm steps)
% Red: 650nm -> channel 26, Green: 550nm -> channel 16, Blue: 450nm -> channel 6
target_channels = [26, 16, 6];
channel_labels = {'Red', 'Green', 'Purple'};

% Data file path template (relative to script location)
data_base_path = fullfile(script_dir, '..', '..', '..', 'exp', 'NTIRE2020_clean');
data_file_name = 'ARAD_HS_0451.mat';

% Wavelength configuration for NTIRE2020_clean (31 channels)
% Based on show_simulation.m, NTIRE2020 uses 31 channels (400-700nm, 10nm steps)
lam31 = [400 410 420 430 440 450 460 470 480 490 500 510 ...
    520 530 540 550 560 570 580 590 600 610 620 630 ...
    640 650 660 670 680 690 700];

% Visualization parameters
intensity = 5;  % Same as show_simulation.m

%% Step 1: Load Reference Method and Select Region of Interest
fprintf('Step 1: Loading reference method data for region selection...\n');

% Use first method as reference
reference_method = methods{reference_method_idx};
fprintf('Using method "%s" as reference for region selection.\n', reference_method);

% Load reference method data
ref_data_file_path = fullfile(data_base_path, reference_method, data_file_name);

if ~exist(ref_data_file_path, 'file')
    error('Reference method file not found: %s\nPlease check the data path.', ref_data_file_path);
end

fprintf('Loading reference data from: %s\n', ref_data_file_path);
ref_hsi_data = load_hsi_data(ref_data_file_path);
[h, w, num_channels] = size(ref_hsi_data);
fprintf('Reference data size: %d x %d x %d channels\n', h, w, num_channels);

% Generate RGB image from hyperspectral data
% Method: Use channels that correspond to Red, Green, Blue wavelengths
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
        rgb_from_hsi(:, :, i) = ref_hsi_data(:, :, ch);
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

% Let user manually select region using getrect (more reliable than imcrop)
fprintf('\nPlease select the region of interest in the image window...\n');
fprintf('Instructions:\n');
fprintf('  1. Click and drag to create a rectangle\n');
fprintf('  2. Release mouse button to finish selection\n');
fprintf('This region will be used for all methods to ensure alignment.\n\n');

figure('Name', 'Select Region of Interest', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
imshow(rgb_from_hsi);
title(sprintf('Reference RGB from %s - Click and drag to select region', reference_method));

% Use getrect which is more straightforward
% getrect returns [x_min, y_min, width, height]
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

% Convert imcrop result [x, y, width, height] format
% getrect returns [x_min, y_min, width, height]
fprintf('Selected crop region: x=%d, y=%d, width=%d, height=%d\n', ...
    rect_crop(1), rect_crop(2), rect_crop(3), rect_crop(4));

%% Save RGB images with red box and cropped region
fprintf('\nGenerating RGB reference images...\n');

% 1. Create image with red bounding box
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
rgb_box_filename = fullfile(output_base_dir, 'reference_RGB_with_red_box.png');
imwrite(rgb_with_box, rgb_box_filename);
fprintf('  Saved: %s\n', rgb_box_filename);

% 2. Crop the selected region
x_start = max(1, rect_crop(1));
y_start = max(1, rect_crop(2));
x_end = min(size(rgb_from_hsi, 2), x_start + rect_crop(3) - 1);
y_end = min(size(rgb_from_hsi, 1), y_start + rect_crop(4) - 1);

rgb_cropped = rgb_from_hsi(y_start:y_end, x_start:x_end, :);

% Save cropped region
rgb_cropped_filename = fullfile(output_base_dir, 'reference_RGB_cropped_region.png');
imwrite(rgb_cropped, rgb_cropped_filename);
fprintf('  Saved: %s\n', rgb_cropped_filename);
fprintf('  Cropped region size: %d x %d\n', size(rgb_cropped, 1), size(rgb_cropped, 2));

% Validate crop region
if rect_crop(3) < 10 || rect_crop(4) < 10
    error('Selected region is too small (width=%d, height=%d). Please select a larger region (at least 10x10 pixels).', ...
        rect_crop(3), rect_crop(4));
end

%% Step 2: Load and Process Data for Each Method
fprintf('\nStep 2: Loading data and generating visualizations...\n');

% Load GT data for difference calculation
fprintf('\nLoading GT data for difference calculation...\n');
gt_data_file_path = fullfile(data_base_path, 'GT', data_file_name);
gt_cropped_data = [];

if exist(gt_data_file_path, 'file')
    try
        gt_hsi_data = load_hsi_data(gt_data_file_path);
        [gt_h, gt_w, gt_num_channels] = size(gt_hsi_data);
        fprintf('GT data size: %d x %d x %d channels\n', gt_h, gt_w, gt_num_channels);
        
        % Crop GT data to the same region
        x_start = max(1, round(rect_crop(1)));
        y_start = max(1, round(rect_crop(2)));
        x_end = min(gt_w, x_start + round(rect_crop(3)) - 1);
        y_end = min(gt_h, y_start + round(rect_crop(4)) - 1);
        
        if x_end >= x_start && y_end >= y_start
            gt_cropped_data = gt_hsi_data(y_start:y_end, x_start:x_end, :);
            fprintf('GT cropped region: %d x %d x %d channels\n', ...
                size(gt_cropped_data, 1), size(gt_cropped_data, 2), size(gt_cropped_data, 3));
        else
            fprintf('Warning: Invalid crop region for GT data\n');
        end
    catch ME_gt
        fprintf('Warning: Could not load GT data: %s\n', ME_gt.message);
        fprintf('Difference heatmaps will not be generated.\n');
    end
else
    fprintf('Warning: GT file not found: %s\n', gt_data_file_path);
    fprintf('Difference heatmaps will not be generated.\n');
end

for method_idx = 1:length(methods)
    method_name = methods{method_idx};
    fprintf('\nProcessing method: %s (%d/%d)\n', method_name, method_idx, length(methods));
    
    % Load data file
    % GT is in the same NTIRE2020_clean directory, just in a subfolder named 'GT'
    data_file_path = fullfile(data_base_path, method_name, data_file_name);
    
    if ~exist(data_file_path, 'file')
        fprintf('Warning: File not found: %s\n', data_file_path);
        fprintf('Skipping method: %s\n', method_name);
        continue;
    end
    
    try
        % Load data using helper function
        hsi_data = load_hsi_data(data_file_path);
        
        % Ensure data is in correct format: [height, width, channels]
        if size(hsi_data, 3) == 1 && (size(hsi_data, 1) == 1 || size(hsi_data, 2) == 1)
            error('Data format error: expected 3D array [H, W, C]');
        end
        
        % Get data dimensions
        [h, w, num_channels] = size(hsi_data);
        fprintf('Loaded data: %d x %d x %d channels\n', h, w, num_channels);
        
        % Check if channels are valid
        if max(target_channels) > num_channels
            fprintf('Warning: Requested channel %d exceeds available channels (%d)\n', ...
                max(target_channels), num_channels);
            continue;
        end
        
        % Crop region
        % rect_crop format: [x_min, y_min, width, height]
        x_start = max(1, round(rect_crop(1)));
        y_start = max(1, round(rect_crop(2)));
        x_end = min(w, x_start + round(rect_crop(3)) - 1);
        y_end = min(h, y_start + round(rect_crop(4)) - 1);
        
        % Ensure valid indices
        if x_end < x_start || y_end < y_start
            fprintf('Warning: Invalid crop region for method %s, skipping...\n', method_name);
            continue;
        end
        
        cropped_data = hsi_data(y_start:y_end, x_start:x_end, :);
        fprintf('Cropped region: %d x %d (from [%d,%d] to [%d,%d])\n', ...
            size(cropped_data, 1), size(cropped_data, 2), x_start, y_start, x_end, y_end);
        
        % Validate cropped data size
        if size(cropped_data, 1) < 10 || size(cropped_data, 2) < 10
            fprintf('Warning: Cropped region too small (%d x %d) for method %s, skipping...\n', ...
                size(cropped_data, 1), size(cropped_data, 2), method_name);
            continue;
        end
        
        % Generate visualization for each target channel
        for ch_idx = 1:length(target_channels)
            channel = target_channels(ch_idx);
            channel_label = channel_labels{ch_idx};
            
            fprintf('  Generating channel %d (%s)...\n', channel, channel_label);
            
            % Extract single channel
            channel_data = cropped_data(:, :, channel);
            
            % Ensure data is in [0, 1] range
            channel_data = double(channel_data);
            channel_data = channel_data - min(channel_data(:));
            if max(channel_data(:)) > 0
                channel_data = channel_data / max(channel_data(:));
            end
            channel_data(channel_data > 1) = 1;
            
            % Get wavelength for this channel
            if channel <= length(lam31)
                wavelength = lam31(channel);
            else
                % Fallback: estimate wavelength (for NTIRE2020: 400 + (channel-1)*10)
                wavelength = 400 + (channel - 1) * 10;
            end
            
            % Validate channel data size
            if size(channel_data, 1) < 10 || size(channel_data, 2) < 10
                fprintf('    Warning: Channel data too small (%d x %d), skipping...\n', ...
                    size(channel_data, 1), size(channel_data, 2));
                continue;
            end
            
            % Prepare data for dispCubeAshwin
            % dispCubeAshwin expects 3D data [H, W, C], so we add a singleton dimension
            channel_data_3d = channel_data;
            channel_data_3d = reshape(channel_data_3d, [size(channel_data, 1), size(channel_data, 2), 1]);
            
            % Ensure data has enough elements for imagedatacube2 (needs at least 50)
            num_elements = numel(channel_data_3d);
            if num_elements < 50
                fprintf('    Warning: Data has only %d elements (need at least 50), skipping...\n', num_elements);
                continue;
            end
            
            % Generate output filename (without extension, dispCubeAshwin adds .png)
            output_filename_base = fullfile(output_base_dir, ...
                sprintf('%s_channel%d_%s', method_name, channel, channel_label));
            
            % Call dispCubeAshwin to generate visualization
            % Parameters: datacube, brightness, wavelength, labels, cols, rows, writefile, grayc, resultname
            try
                % Add path to dispCubeAshwin if needed
                addpath('..');
                
                % Ensure output directory exists
                [output_dir, ~, ~] = fileparts(output_filename_base);
                if ~exist(output_dir, 'dir')
                    mkdir(output_dir);
                end
                
                % Full path with extension
                output_full_path = [output_filename_base, '.png'];
                fprintf('    Generating: %s\n', output_full_path);
                
                % Call dispCubeAshwin - it creates and saves the figure
                dispCubeAshwin(channel_data_3d, intensity, wavelength, [], 1, 1, 0, 1, output_filename_base);
                
                % Wait a moment for figure to be created
                pause(0.3);
                
                % Get the figure created by dispCubeAshwin
                fig_handles = findall(0,'Type','figure');
                if ~isempty(fig_handles)
                    current_fig = fig_handles(1);
                    
                    % Get all axes in the figure
                    axes_handles = findall(current_fig, 'Type', 'axes');
                    
                    % Method 3: Adjust Position to remove white borders
                    % This eliminates white space by changing the axes Position
                    for ax_idx = 1:length(axes_handles)
                        ax = axes_handles(ax_idx);
                        % Set Position to eliminate white borders
                        set(ax, 'Position', get(ax, 'OuterPosition') - ...
                            get(ax, 'TightInset') * [-1 0 1 0; 0 -1 0 1; 0 0 1 0; 0 0 0 1]);
                    end
                    
                    % Also set LooseInset to zero as backup
                    for ax_idx = 1:length(axes_handles)
                        set(axes_handles(ax_idx), 'LooseInset', [0, 0, 0, 0]);
                    end
                    
                    % Set figure properties
                    set(current_fig, 'PaperPositionMode', 'auto');
                    
                    % Force a redraw to apply changes
                    drawnow;
                    pause(0.1);
                    
                    % Use getframe to capture the figure content
                    frame = getframe(current_fig);
                    img_frame = frame.cdata;
                    
                    % Save using imwrite (same method as reference RGB images)
                    imwrite(img_frame, output_full_path);
                    
                    fprintf('    ✓ Saved (no white borders, using Position adjustment): %s\n', output_full_path);
                else
                    fprintf('    ✗ Warning: No figure found after dispCubeAshwin\n');
                end
                
                % Close the figure after saving
                close all;
                rmpath('..');
                
            catch ME
                % Make sure to close figure even if there's an error
                if ~isempty(findall(0,'Type','figure'))
                    close all;
                end
                fprintf('    ✗ Error generating visualization: %s\n', ME.message);
                fprintf('    Stack trace:\n');
                for k = 1:length(ME.stack)
                    fprintf('      %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
                end
                rmpath('..');
            end
        end
        
        % Generate difference heatmap (skip GT)
        if ~strcmp(method_name, 'GT') && ~isempty(gt_cropped_data)
            try
                fprintf('  Generating difference heatmap vs GT...\n');
                
                % Check if dimensions match
                if ~isequal(size(cropped_data), size(gt_cropped_data))
                    fprintf('    Warning: Dimension mismatch between %s and GT, skipping heatmap\n', method_name);
                    fprintf('    Method: %d x %d x %d, GT: %d x %d x %d\n', ...
                        size(cropped_data, 1), size(cropped_data, 2), size(cropped_data, 3), ...
                        size(gt_cropped_data, 1), size(gt_cropped_data, 2), size(gt_cropped_data, 3));
                else
                    % Calculate squared difference
                    diff_squared = (double(cropped_data) - double(gt_cropped_data)).^2;
                    
                    % Average across all channels
                    diff_mean = mean(diff_squared, 3);
                    
                    % Normalize to [0, 1] range
                    diff_min = min(diff_mean(:));
                    diff_max = max(diff_mean(:));
                    if diff_max > diff_min
                        diff_normalized = (diff_mean - diff_min) / (diff_max - diff_min);
                    else
                        diff_normalized = zeros(size(diff_mean));
                    end
                    
                    % Create figure for heatmap
                    fig_heatmap = figure('Visible', 'off', 'Color', 'white');
                    ax = axes('Parent', fig_heatmap);
                    
                    % Create custom blue-to-red colormap
                    % Blue (low values, difference = 0) -> Red (high values, difference = 1)
                    n_colors = 256;
                    % Method 1: Create explicit RGB colormap
                    % Blue: [0, 0, 1] -> Red: [1, 0, 0]
                    red_blue_map = zeros(n_colors, 3);
                    for i = 1:n_colors
                        t = (i - 1) / (n_colors - 1);  % t from 0 to 1
                        red_blue_map(i, 1) = t;         % Red: 0 -> 1
                        red_blue_map(i, 2) = 0;         % Green: always 0
                        red_blue_map(i, 3) = 1 - t;     % Blue: 1 -> 0
                    end
                    
                    % Display heatmap
                    imagesc(ax, diff_normalized);
                    colormap(ax, red_blue_map);
                    axis(ax, 'image');
                    axis(ax, 'off');
                    
                    % Set color limits explicitly to ensure proper mapping
                    caxis(ax, [0, 1]);
                    
                    % Add colorbar with label
                    cb = colorbar(ax);
                    ylabel(cb, 'Difference (Normalized)', 'FontSize', 10);
                    
                    % Ensure colormap is applied
                    set(fig_heatmap, 'Colormap', red_blue_map);
                    
                    % Remove white borders using same method as channel images
                    set(ax, 'Position', get(ax, 'OuterPosition') - ...
                        get(ax, 'TightInset') * [-1 0 1 0; 0 -1 0 1; 0 0 1 0; 0 0 0 1]);
                    set(ax, 'LooseInset', [0, 0, 0, 0]);
                    set(fig_heatmap, 'PaperPositionMode', 'auto');
                    
                    % Force redraw to ensure colormap is applied
                    drawnow;
                    pause(0.2);
                    
                    % Save heatmap
                    output_heatmap_path = fullfile(output_base_dir, ...
                        sprintf('%s_diff_heatmap.png', method_name));
                    
                    % Convert normalized difference to RGB using colormap
                    % Map [0, 1] to [1, n_colors] for colormap indexing
                    diff_indexed = round(diff_normalized * (n_colors - 1)) + 1;
                    diff_indexed = max(1, min(n_colors, diff_indexed));  % Clamp to valid range
                    
                    % Convert indexed image to RGB using our red-blue colormap
                    diff_rgb = ind2rgb(diff_indexed, red_blue_map);
                    
                    % Convert to uint8 for saving
                    diff_rgb_uint8 = uint8(diff_rgb * 255);
                    
                    % Save the RGB image directly
                    imwrite(diff_rgb_uint8, output_heatmap_path);
                    
                    fprintf('    ✓ Saved difference heatmap: %s\n', output_heatmap_path);
                    
                    % Close figure
                    close(fig_heatmap);
                end
            catch ME_heatmap
                fprintf('    ✗ Error generating difference heatmap: %s\n', ME_heatmap.message);
                if ~isempty(findall(0,'Type','figure'))
                    close all;
                end
            end
        end
        
    catch ME
        fprintf('Error processing method %s: %s\n', method_name, ME.message);
        continue;
    end
end

fprintf('\n=== Generation Complete ===\n');
fprintf('Output directory: %s\n', output_base_dir);
fprintf('Total methods processed: %d (including GT)\n', length(methods));
num_channel_images = length(methods) * length(target_channels);
num_diff_images = length(methods) - 1;  % All methods except GT
num_rgb_images = 2;
fprintf('Total images generated:\n');
fprintf('  - Channel images: %d (methods) x %d (channels) = %d\n', ...
    length(methods), length(target_channels), num_channel_images);
fprintf('  - Difference heatmaps: %d (methods excluding GT) = %d\n', ...
    length(methods) - 1, num_diff_images);
fprintf('  - RGB reference images: %d\n', num_rgb_images);
fprintf('  - Total: %d images\n', num_channel_images + num_diff_images + num_rgb_images);

% Close any remaining figures
close all;

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
    fprintf('  File size: %d bytes\n', file_size);
    
    % Check file format by reading first few bytes
    fid = fopen(data_file_path, 'r');
    if fid ~= -1
        first_bytes = fread(fid, min(16, file_size), 'uint8');
        fclose(fid);
        % MATLAB v7.3 files start with HDF5 signature: 0x89 0x48 0x44 0x46 0x0D 0x0A 0x1A 0x0A
        % Standard MAT files start with 'MATLAB' or specific header
        if length(first_bytes) >= 8
            if all(first_bytes(1:8) == [137; 72; 68; 70; 13; 10; 26; 10])  % HDF5 signature
                fprintf('  File format: HDF5 (MATLAB v7.3)\n');
            elseif length(first_bytes) >= 6 && all(first_bytes(1:6)' == double('MATLAB'))
                fprintf('  File format: Standard MATLAB binary\n');
            else
                fprintf('  Warning: File does not appear to be a valid MATLAB file\n');
                fprintf('  First bytes (hex): %s\n', sprintf('%02X ', first_bytes(1:min(8, length(first_bytes)))));
            end
        end
    end
    
    if file_size < 1000  % Less than 1KB is suspicious
        fprintf('  Warning: File size is very small (%d bytes), file might be corrupted\n', file_size);
        % Try to restore from backup if exists
        backup_path = [data_file_path '.backup'];
        if exist(backup_path, 'file')
            backup_info = dir(backup_path);
            fprintf('  Found backup file (size: %d bytes), attempting to restore...\n', backup_info.bytes);
            copyfile(backup_path, data_file_path);
            file_info = dir(data_file_path);
            file_size = file_info.bytes;
            fprintf('  Restored file size: %d bytes\n', file_size);
        end
    end
    
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
                            % Try to detect and transpose if needed
                            if ndims(hsi_data) == 3
                                [d1, d2, d3] = size(hsi_data);
                                % If first dimension is small (likely channels)
                                if d1 < 50 && d2 > 100 && d3 > 100
                                    % Likely [C, H, W] -> need [H, W, C]
                                    hsi_data = permute(hsi_data, [2, 3, 1]);
                                elseif d3 < 50 && d1 > 100 && d2 > 100
                                    % Likely [H, W, C] already
                                    % No change needed
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
                % Method 4: Try to convert HDF5 to standard MAT format using Python
                fprintf('  Attempting to convert HDF5 file using Python...\n');
                try
                    % Get the directory of the current script
                    st = dbstack('-completenames');
                    if length(st) > 1
                        script_path = st(end).file;
                        script_dir = fileparts(script_path);
                    else
                        script_dir = pwd;
                    end
                    
                    python_script = fullfile(script_dir, 'convert_hdf5_to_mat.py');
                    
                    % Check if Python script exists
                    if ~exist(python_script, 'file')
                        error('Python conversion script not found: %s', python_script);
                    end
                    
                    % Call Python script to convert the file
                    python_cmd = sprintf('python "%s" "%s"', python_script, data_file_path);
                    fprintf('  Running: %s\n', python_cmd);
                    [status, cmdout] = system(python_cmd);
                    
                    if status == 0
                        fprintf('  Conversion successful, retrying load...\n');
                        % Try loading again after conversion
                        data_struct = load(data_file_path);
                        if isempty(fieldnames(data_struct))
                            error('Converted file is empty');
                        end
                    else
                        error('Python conversion failed: %s', cmdout);
                    end
                catch ME_python
                    % All methods failed - try one more thing: check if backup exists and try it
                    backup_path = [data_file_path '.backup'];
                    if exist(backup_path, 'file')
                        fprintf('  All methods failed for original file.\n');
                        fprintf('  Trying backup file: %s\n', backup_path);
                        try
                            % Try loading backup file
                            data_struct = load(backup_path);
                            if ~isempty(fieldnames(data_struct))
                                fprintf('  Successfully loaded from backup! Copying to original location...\n');
                                copyfile(backup_path, data_file_path);
                                % Now try to load the restored file
                                if isfield(data_struct, 'cube')
                                    hsi_data = data_struct.cube;
                                else
                                    field_names = fieldnames(data_struct);
                                    for i = 1:length(field_names)
                                        var_name = field_names{i};
                                        if ~strcmp(var_name(1), '_')
                                            var_data = data_struct.(var_name);
                                            if ndims(var_data) == 3 || ndims(var_data) == 4
                                                hsi_data = var_data;
                                                if ndims(var_data) == 4
                                                    hsi_data = squeeze(hsi_data(1, :, :, :));
                                                end
                                                break;
                                            end
                                        end
                                    end
                                end
                                if ~isempty(hsi_data)
                                    fprintf('  Successfully restored and loaded from backup!\n');
                                    return;
                                end
                            end
                        catch ME_backup
                            fprintf('  Backup file also failed: %s\n', ME_backup.message);
                        end
                    end
                    
                    % All methods failed - provide helpful message
                    fprintf('  All loading methods failed.\n');
                    fprintf('  Standard load: %s\n', ME_load.message);
                    fprintf('  Matfile: %s\n', ME_matfile.message);
                    fprintf('  H5read: %s\n', ME_h5.message);
                    fprintf('  Python conversion: %s\n', ME_python.message);
                    fprintf('\n  SUGGESTION: The file %s appears to be corrupted.\n', data_file_path);
                    fprintf('  You may need to:\n');
                    fprintf('  1. Regenerate this file from the original source\n');
                    fprintf('  2. Check if other files in the same directory can be read (e.g., ARAD_HS_0451.mat)\n');
                    fprintf('  3. Verify the file was not corrupted during transfer\n');
                    error('Could not read file: %s\nFile appears to be corrupted or in an unsupported format.', ...
                        data_file_path);
                end
            end
        end
    end
    
    if ~use_h5py && isempty(hsi_data)
        % Detect variable name (cube or pred)
        if isfield(data_struct, 'cube')
            hsi_data = data_struct.cube;
        elseif isfield(data_struct, 'pred')
            pred_data = data_struct.pred;
            % Handle pred format (similar to show_real.m)
            if ndims(pred_data) == 4
                % If 4D, take first frame and squeeze
                hsi_data = squeeze(pred_data(1, :, :, :));
            else
                hsi_data = pred_data;
            end
            % Apply flip if needed (based on show_real.m)
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
