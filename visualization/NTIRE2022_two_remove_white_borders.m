%% Remove White Borders from Images
% This script removes large white border areas from images in the specified directory
% It only removes white borders at the edges, not individual white pixels within the image

clear; clc;
close all;

%% Configuration
% Get script directory
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

% Default: process images in comparison_results/ARAD_1K_0924
input_dir = fullfile(script_dir, 'comparison_results', 'ARAD_1K_0924');

% If you want to specify a different directory, uncomment and modify:
% input_dir = uigetdir(script_dir, 'Select directory containing images to process');

if ~exist(input_dir, 'dir')
    error('Directory not found: %s', input_dir);
end

fprintf('Processing images in: %s\n', input_dir);

%% Get all PNG files
png_files = dir(fullfile(input_dir, '*.png'));
num_files = length(png_files);

if num_files == 0
    error('No PNG files found in directory: %s', input_dir);
end

fprintf('Found %d PNG files to process.\n\n', num_files);

%% Process each image
processed_count = 0;
skipped_count = 0;

for i = 1:num_files
    file_path = fullfile(input_dir, png_files(i).name);
    fprintf('Processing [%d/%d]: %s\n', i, num_files, png_files(i).name);
    
    try
        % Read the image
        img = imread(file_path);
        original_size = size(img);
        
        % Convert to double for processing
        if size(img, 3) == 3
            % RGB image
            img_double = double(img) / 255.0;
        else
            % Grayscale image, convert to RGB
            img_double = repmat(double(img) / 255.0, [1, 1, 3]);
        end
        
        % Find white border regions (large continuous white areas at edges)
        [img_cropped, crop_info] = remove_white_borders_from_image(img_double);
        
        % Check if cropping made a difference
        if isequal(size(img_cropped), original_size(1:2))
            fprintf('  No white borders detected, skipping.\n');
            skipped_count = skipped_count + 1;
        else
            % Convert back to uint8 if needed
            if max(img_cropped(:)) <= 1
                img_cropped = uint8(img_cropped * 255);
            else
                img_cropped = uint8(img_cropped);
            end
            
            % Save the cropped image (overwrite original)
            imwrite(img_cropped, file_path);
            
            fprintf('  Cropped: %dx%d -> %dx%d (removed %d rows top, %d rows bottom, %d cols left, %d cols right)\n', ...
                original_size(2), original_size(1), ...
                size(img_cropped, 2), size(img_cropped, 1), ...
                crop_info.top, crop_info.bottom, crop_info.left, crop_info.right);
            
            processed_count = processed_count + 1;
        end
        
    catch ME
        fprintf('  Error processing %s: %s\n', png_files(i).name, ME.message);
    end
end

fprintf('\n=== Processing Complete ===\n');
fprintf('Total files: %d\n', num_files);
fprintf('Processed (cropped): %d\n', processed_count);
fprintf('Skipped (no borders): %d\n', skipped_count);

%% Helper Function: Remove White Borders from Image
function [img_cropped, crop_info] = remove_white_borders_from_image(img)
    % Remove large white border areas from image edges
    % img: input image (double, range [0, 1])
    % Returns: cropped image and crop information
    
    [h, w, ~] = size(img);
    
    % Threshold for white (consider pixels with all channels > threshold as white)
    white_threshold = 0.95;  % Adjust this if needed (0.95 = 95% white)
    
    % Minimum border width to remove (to avoid removing single pixel lines)
    min_border_width = 5;  % Remove borders only if they are at least 5 pixels wide
    
    % Convert to grayscale for white detection
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    
    % Find white pixels (value > threshold)
    white_mask = gray > white_threshold;
    
    % Scan from top edge
    top_remove = 0;
    for row = 1:min(h, 100)  % Check up to 100 rows from top
        white_ratio = sum(white_mask(row, :)) / w;
        if white_ratio > 0.9  % If 90% of the row is white
            top_remove = row;
        else
            break;  % Stop when we find a non-white row
        end
    end
    
    % Only remove if it's a large border (at least min_border_width pixels)
    if top_remove < min_border_width
        top_remove = 0;
    end
    
    % Scan from bottom edge
    bottom_remove = 0;
    for row = h:-1:max(1, h-100)  % Check up to 100 rows from bottom
        white_ratio = sum(white_mask(row, :)) / w;
        if white_ratio > 0.9  % If 90% of the row is white
            bottom_remove = h - row + 1;
        else
            break;  % Stop when we find a non-white row
        end
    end
    
    % Only remove if it's a large border
    if bottom_remove < min_border_width
        bottom_remove = 0;
    end
    
    % Scan from left edge
    left_remove = 0;
    for col = 1:min(w, 100)  % Check up to 100 columns from left
        white_ratio = sum(white_mask(:, col)) / h;
        if white_ratio > 0.9  % If 90% of the column is white
            left_remove = col;
        else
            break;  % Stop when we find a non-white column
        end
    end
    
    % Only remove if it's a large border
    if left_remove < min_border_width
        left_remove = 0;
    end
    
    % Scan from right edge
    right_remove = 0;
    for col = w:-1:max(1, w-100)  % Check up to 100 columns from right
        white_ratio = sum(white_mask(:, col)) / h;
        if white_ratio > 0.9  % If 90% of the column is white
            right_remove = w - col + 1;
        else
            break;  % Stop when we find a non-white column
        end
    end
    
    % Only remove if it's a large border
    if right_remove < min_border_width
        right_remove = 0;
    end
    
    % Crop the image
    row_start = top_remove + 1;
    row_end = h - bottom_remove;
    col_start = left_remove + 1;
    col_end = w - right_remove;
    
    % Ensure valid indices
    row_start = max(1, row_start);
    row_end = min(h, row_end);
    col_start = max(1, col_start);
    col_end = min(w, col_end);
    
    if row_end < row_start || col_end < col_start
        % Invalid crop, return original
        img_cropped = img;
        crop_info.top = 0;
        crop_info.bottom = 0;
        crop_info.left = 0;
        crop_info.right = 0;
    else
        img_cropped = img(row_start:row_end, col_start:col_end, :);
        crop_info.top = top_remove;
        crop_info.bottom = bottom_remove;
        crop_info.left = left_remove;
        crop_info.right = right_remove;
    end
end
