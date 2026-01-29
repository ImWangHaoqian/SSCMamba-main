"""
Convert HDF5 format .mat files to standard MATLAB .mat format
This script converts HDF5 v7.3 files to v7 format that MATLAB can read
"""

import sys
import os
import scipy.io
import numpy as np

# Try to import hdf5storage, but handle import errors gracefully
HAS_HDF5STORAGE = False
try:
    import hdf5storage
    HAS_HDF5STORAGE = True
except (ImportError, ValueError):
    # Handle both ImportError and ValueError (numpy compatibility issues)
    HAS_HDF5STORAGE = False
    print('Warning: hdf5storage not available, trying alternative methods')


def convert_hdf5_to_mat(hdf5_path, output_path=None):
    """
    Convert HDF5 format .mat file to standard MATLAB .mat format

    Args:
        hdf5_path: Path to input HDF5 .mat file
        output_path: Path to output .mat file (if None, overwrites input)
    """
    if output_path is None:
        # Create backup and convert in place
        backup_path = hdf5_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(hdf5_path, backup_path)
            print(f'Created backup: {backup_path}')
        output_path = hdf5_path

    print(f'Reading file: {hdf5_path}')

    # Method 1: Try scipy.io.loadmat first (most compatible)
    data_dict = None
    e1_msg = None
    try:
        print('  Trying scipy.io.loadmat...')
        data_dict = scipy.io.loadmat(hdf5_path)
        # Remove MATLAB metadata
        data_dict = {k: v for k, v in data_dict.items()
                     if not k.startswith('__')}
        if data_dict:
            print(f'  Success! Found keys: {list(data_dict.keys())}')
        else:
            raise ValueError('No data found')
    except Exception as e1:
        e1_msg = str(e1)
        print(f'  scipy.io.loadmat failed: {e1_msg}')

    # Method 2: Try using hdf5storage (designed for MATLAB v7.3 files)
    e2_msg = None
    if data_dict is None and HAS_HDF5STORAGE:
        try:
            print('  Trying hdf5storage...')
            data_dict = hdf5storage.loadmat(hdf5_path)
            # Remove MATLAB metadata
            data_dict = {k: v for k, v in data_dict.items()
                         if not k.startswith('__')}
            print(f'  Success! Found keys: {list(data_dict.keys())}')
        except Exception as e2:
            e2_msg = str(e2)
            print(f'  hdf5storage failed: {e2_msg}')

    # Method 3: Try h5py as last resort
    e3_msg = None
    if data_dict is None:
        try:
            import h5py
            print('  Trying h5py...')

            def get_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dataset = np.array(obj[()])
                    key = name.split('/')[-1] if '/' in name else name
                    if not key.startswith('#'):
                        data_dict[key] = dataset

            with h5py.File(hdf5_path, 'r') as f:
                data_dict = {}
                f.visititems(get_datasets)
                if not data_dict:
                    raise ValueError('No datasets found')
                print(f'  Success! Found datasets: '
                      f'{list(data_dict.keys())}')
        except Exception as e3:
            e3_msg = str(e3)
            print(f'  h5py failed: {e3_msg}')
            # All methods failed
            error_msg = (
                f'All methods failed:\n'
                f'  scipy.io: {e1_msg}\n'
                f'  hdf5storage: {e2_msg or "N/A (not available)"}\n'
                f'  h5py: {e3_msg}')
            raise ValueError(error_msg)

    # Save as standard MATLAB .mat format (v7)
    print(f'Saving as standard MAT file: {output_path}')
    scipy.io.savemat(output_path, data_dict, format='7')
    print(f'Conversion complete: {output_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python convert_hdf5_to_mat.py <input_file.mat>')
        print('Example: python convert_hdf5_to_mat.py ARAD_1K_0924.mat')
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f'Error: File not found: {input_file}')
        sys.exit(1)

    try:
        convert_hdf5_to_mat(input_file, output_file)
        print('Success!')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
