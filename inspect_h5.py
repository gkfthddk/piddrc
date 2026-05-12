#!/usr/bin/env python
"""Quickly inspect the schema, keys, and shapes of an HDF5 file."""

import argparse
import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name:40} | Shape: {str(obj.shape):20} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group:   {name}")

def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 file schema.")
    parser.add_argument("file", help="Path to the HDF5 file.")
    args = parser.parse_args()

    try:
        with h5py.File(args.file, "r") as f:
            print(f"--- Schema for {args.file} ---")
            f.visititems(print_structure)
    except Exception as e:
        print(f"Failed to read {args.file}: {e}")

if __name__ == "__main__":
    main()
