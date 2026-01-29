#!/usr/bin/env python3
import h5py
import numpy as np
import os
import argparse
import sys

def inspect_h5(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at: {file_path}")
        sys.exit(1)

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Opening: {file_path}")
            print(f"Total Videos stored: {len(f.keys())}\n")
            print("-" * 50)

            # Iterate over every video group in the file
            for video_name in f.keys():
                print(f"Video Group: {video_name}")
                group = f[video_name]
                
                # Iterate over every dataset (feature) in that group
                for key in group.keys():
                    data = group[key][:]
                    
                    if isinstance(data, np.ndarray):
                        print(f"   ├── Key: '{key}'")
                        print(f"   │   ├── Shape: {data.shape}")
                        print(f"   │   ├── Type:  {data.dtype}")
                        
                        # Print sample stats
                        if data.size > 0:
                            print(f"   │   └── Mean Value: {np.mean(data):.5f}")
                        else:
                            print(f"   │   └── Empty Data")
                    else:
                        print(f"   ├── Key: '{key}' -> Value: {data}")
                
                print("-" * 50) 
                
    except OSError:
        print(f"Error: Could not open file. Is '{file_path}' a valid HDF5 file?")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # parser setup
    parser = argparse.ArgumentParser(
        description="Inspect the structure and contents of a Video Features HDF5 file."
    )
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the .h5 file you want to inspect"
    )

    args = parser.parse_args()
    inspect_h5(args.file_path)