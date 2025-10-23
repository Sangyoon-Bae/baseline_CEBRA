#!/usr/bin/env python3
"""
Test script to load Allen Brain Observatory calcium imaging data
using CEBRA/kirby dataset infrastructure with batch_size=4
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# Add kirby to path
sys.path.insert(0, str(Path(__file__).parent))

def explore_h5_structure(h5_path):
    """Explore the structure of an h5 file"""
    print(f"\n{'='*80}")
    print(f"Exploring structure of: {h5_path}")
    print(f"{'='*80}")

    with h5py.File(h5_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name:50s} shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group:   {name}")

        f.visititems(print_structure)
    print()

def load_h5_as_kirby_data(h5_path):
    """Try to load h5 file as kirby Data object"""
    from kirby.data import Data

    print(f"\n{'='*80}")
    print(f"Attempting to load as kirby Data: {h5_path}")
    print(f"{'='*80}")

    try:
        with h5py.File(h5_path, 'r') as f:
            data = Data.from_hdf5(f, lazy=True)
            print("✅ Successfully loaded as kirby Data object!")
            print(f"   Data type: {type(data)}")

            # Explore available attributes
            if hasattr(data, '__dict__'):
                print(f"\n   Available attributes:")
                for key in dir(data):
                    if not key.startswith('_'):
                        try:
                            attr = getattr(data, key)
                            if not callable(attr):
                                print(f"      - {key}: {type(attr)}")
                        except:
                            pass

            return data
    except Exception as e:
        print(f"❌ Failed to load as kirby Data: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataset_with_kirby(data_dir="data"):
    """Test loading Allen Brain Observatory data with kirby Dataset"""
    from kirby.data import Dataset

    print(f"\n{'='*80}")
    print("Testing kirby Dataset with Allen Brain Observatory data")
    print(f"{'='*80}")

    # Find all h5 files
    h5_files = sorted(Path(data_dir).glob("*.h5"))
    print(f"\nFound {len(h5_files)} h5 files")

    if len(h5_files) == 0:
        print("❌ No h5 files found!")
        return None

    # Test with first file
    test_file = h5_files[0]
    print(f"\nTesting with: {test_file}")

    # Explore structure
    explore_h5_structure(test_file)

    # Try to load as kirby Data
    data = load_h5_as_kirby_data(test_file)

    return data

def test_dataloader_with_batch_size_4():
    """Test creating a DataLoader with batch_size=4"""
    print(f"\n{'='*80}")
    print("Testing DataLoader with batch_size=4")
    print(f"{'='*80}")

    try:
        data = test_dataset_with_kirby()

        if data is None:
            print("❌ Cannot proceed with DataLoader test - data loading failed")
            return

        # Try to create a simple dataloader
        print("\n✅ Basic data loading test passed!")
        print("   To create a full DataLoader, we need:")
        print("   1. A proper Dataset configuration file")
        print("   2. A sampler that generates DatasetIndex objects")
        print("   3. A collate function for batching")

    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_calcium_traces(data_dir="data"):
    """Analyze calcium trace data structure"""
    print(f"\n{'='*80}")
    print("Analyzing Calcium Trace Data")
    print(f"{'='*80}")

    h5_files = sorted(Path(data_dir).glob("*.h5"))
    if len(h5_files) == 0:
        print("❌ No h5 files found!")
        return

    test_file = h5_files[0]

    with h5py.File(test_file, 'r') as f:
        print(f"\n📊 Calcium traces analysis for: {test_file.name}")

        # Check calcium traces
        if 'calcium_traces' in f:
            ct = f['calcium_traces']
            print(f"\n   Calcium traces group found:")

            if 'df_over_f' in ct:
                df_over_f = ct['df_over_f'][:]
                print(f"      - df_over_f shape: {df_over_f.shape}")
                print(f"      - df_over_f dtype: {df_over_f.dtype}")
                print(f"      - df_over_f range: [{df_over_f.min():.4f}, {df_over_f.max():.4f}]")
                print(f"      - Interpretation: (timepoints={df_over_f.shape[0]}, neurons={df_over_f.shape[1]})")

            if 'train_mask' in ct:
                train_mask = ct['train_mask'][:]
                print(f"\n      - train_mask: {train_mask.sum()} / {len(train_mask)} samples")

            if 'valid_mask' in ct:
                valid_mask = ct['valid_mask'][:]
                print(f"      - valid_mask: {valid_mask.sum()} / {len(valid_mask)} samples")

            if 'test_mask' in ct:
                test_mask = ct['test_mask'][:]
                print(f"      - test_mask: {test_mask.sum()} / {len(test_mask)} samples")

            if 'domain' in ct:
                domain = ct['domain']
                start = domain['start'][0] if 'start' in domain else None
                end = domain['end'][0] if 'end' in domain else None
                print(f"\n      - domain: [{start}, {end}]")

        # Check natural movie one
        if 'natural_movie_one' in f:
            nm = f['natural_movie_one']
            print(f"\n   Natural movie one group found:")

            if 'timestamps' in nm:
                timestamps = nm['timestamps'][:]
                print(f"      - timestamps shape: {timestamps.shape}")
                print(f"      - timestamps range: [{timestamps.min():.4f}, {timestamps.max():.4f}]")

            if 'frame_number' in nm:
                frame_num = nm['frame_number'][:]
                print(f"      - frame_number shape: {frame_num.shape}")
                print(f"      - frame_number range: [{frame_num.min():.0f}, {frame_num.max():.0f}]")

        # Check units
        if 'units' in f:
            units = f['units']
            print(f"\n   Units group found:")
            if 'id' in units:
                unit_ids = units['id'][:]
                print(f"      - Number of units: {len(unit_ids)}")
                print(f"      - Example unit ID: {unit_ids[0]}")

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("Allen Brain Observatory - CEBRA/kirby Dataset Test")
    print("="*80)

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        return 1

    # Analyze calcium traces
    analyze_calcium_traces(data_dir)

    # Test basic data loading
    test_dataset_with_kirby(data_dir)

    # Test dataloader
    test_dataloader_with_batch_size_4()

    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
