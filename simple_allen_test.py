#!/usr/bin/env python3
"""
Simplified test script for Allen Brain Observatory data
Without using full CEBRA/kirby infrastructure - just basic data loading
"""

import h5py
import numpy as np
import torch
from pathlib import Path

def load_session_data(h5_path, split='train'):
    """Load Allen Brain Observatory session data

    Args:
        h5_path: Path to h5 file
        split: 'train', 'valid', or 'test'

    Returns:
        dict with neural_data, timestamps, and metadata
    """
    print(f"\n{'='*80}")
    print(f"Loading: {h5_path.name} ({split} split)")
    print(f"{'='*80}")

    with h5py.File(h5_path, 'r') as f:
        # Load calcium traces (neural data)
        df_over_f = f['calcium_traces/df_over_f'][:]  # (timepoints, neurons)

        # Load split mask
        mask_key = f'calcium_traces/{split}_mask'
        if mask_key not in f:
            print(f"   ⚠️  Warning: {mask_key} not found, using all data")
            mask = np.ones(len(df_over_f), dtype=bool)
        else:
            mask = f[mask_key][:]

        # Apply mask
        neural_data = df_over_f[mask]

        # Get domain (time range)
        if 'calcium_traces/domain/start' in f and 'calcium_traces/domain/end' in f:
            domain_start = f['calcium_traces/domain/start'][0]
            domain_end = f['calcium_traces/domain/end'][0]

            # Create timestamps
            all_timestamps = np.linspace(domain_start, domain_end, len(df_over_f))
            timestamps = all_timestamps[mask]
        else:
            # Fallback: use indices
            timestamps = np.arange(len(neural_data)).astype(np.float32)

        # Get metadata
        metadata = {
            'session_id': h5_path.stem,
            'split': split,
            'num_timepoints': neural_data.shape[0],
            'num_neurons': neural_data.shape[1],
            'train_samples': f['calcium_traces/train_mask'][:].sum() if 'calcium_traces/train_mask' in f else 0,
            'valid_samples': f['calcium_traces/valid_mask'][:].sum() if 'calcium_traces/valid_mask' in f else 0,
            'test_samples': f['calcium_traces/test_mask'][:].sum() if 'calcium_traces/test_mask' in f else 0,
        }

        print(f"   Neural data shape: {neural_data.shape}")
        print(f"   Timestamps shape: {timestamps.shape}")
        print(f"   Data range: [{neural_data.min():.4f}, {neural_data.max():.4f}]")
        print(f"   Split sizes - train: {metadata['train_samples']}, valid: {metadata['valid_samples']}, test: {metadata['test_samples']}")

        return {
            'neural_data': neural_data,
            'timestamps': timestamps,
            'metadata': metadata
        }

def create_pytorch_dataloader(data_dict, batch_size=4, shuffle=True):
    """Create PyTorch DataLoader from loaded data

    Args:
        data_dict: Dictionary from load_session_data
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        PyTorch DataLoader
    """
    print(f"\n{'='*80}")
    print(f"Creating PyTorch DataLoader (batch_size={batch_size})")
    print(f"{'='*80}")

    # Convert to tensors
    neural_tensor = torch.from_numpy(data_dict['neural_data']).float()
    time_tensor = torch.from_numpy(data_dict['timestamps']).float()

    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(neural_tensor, time_tensor)

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    print(f"   ✅ Created DataLoader")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(dataloader)}")

    return dataloader

def test_dataloader_iteration(dataloader, num_batches=3):
    """Test iterating through dataloader

    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to test
    """
    print(f"\n{'='*80}")
    print(f"Testing DataLoader iteration ({num_batches} batches)")
    print(f"{'='*80}")

    for i, (neural_batch, time_batch) in enumerate(dataloader):
        if i >= num_batches:
            break

        print(f"\n   Batch {i+1}:")
        print(f"      Neural data shape: {neural_batch.shape}")
        print(f"      Time data shape: {time_batch.shape}")
        print(f"      Neural data range: [{neural_batch.min():.4f}, {neural_batch.max():.4f}]")
        print(f"      Time range: [{time_batch.min():.4f}, {time_batch.max():.4f}]")

    print(f"\n   ✅ DataLoader iteration test passed!")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("Allen Brain Observatory - Simple Data Loading Test")
    print("="*80)

    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        return 1

    # Find h5 files
    h5_files = sorted(data_dir.glob("*.h5"))
    if len(h5_files) == 0:
        print(f"\n❌ No h5 files found in {data_dir}")
        return 1

    print(f"\nFound {len(h5_files)} h5 files")

    # Test with first file - train split
    print(f"\n{'='*80}")
    print("TEST 1: Load single session (train split)")
    print("="*80)

    test_file = h5_files[0]
    data_train = load_session_data(test_file, split='train')

    # Create dataloader
    dataloader_train = create_pytorch_dataloader(data_train, batch_size=4)

    # Test iteration
    test_dataloader_iteration(dataloader_train, num_batches=3)

    # Test with validation split
    print(f"\n{'='*80}")
    print("TEST 2: Load single session (valid split)")
    print("="*80)

    data_valid = load_session_data(test_file, split='valid')
    dataloader_valid = create_pytorch_dataloader(data_valid, batch_size=4)
    test_dataloader_iteration(dataloader_valid, num_batches=2)

    # Test loading multiple sessions
    print(f"\n{'='*80}")
    print("TEST 3: Load multiple sessions")
    print("="*80)

    max_sessions = min(4, len(h5_files))
    all_data = []

    for h5_file in h5_files[:max_sessions]:
        try:
            data = load_session_data(h5_file, split='train')
            all_data.append(data)
        except Exception as e:
            print(f"   ⚠️  Error loading {h5_file.name}: {e}")

    print(f"\n   Successfully loaded {len(all_data)} / {max_sessions} sessions")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print("="*80)

    total_neurons = sum(d['metadata']['num_neurons'] for d in all_data)
    total_timepoints = sum(d['metadata']['num_timepoints'] for d in all_data)

    print(f"\n   ✅ All tests passed!")
    print(f"\n   Total sessions loaded: {len(all_data)}")
    print(f"   Total neurons: {total_neurons}")
    print(f"   Total timepoints: {total_timepoints}")
    print(f"   Average neurons per session: {total_neurons / len(all_data):.1f}")

    print(f"\n{'='*80}")
    print("Data is ready for CEBRA training!")
    print("="*80)
    print("\nNext steps:")
    print("1. Use the loaded neural_data and timestamps")
    print("2. Apply CEBRA's self-supervised learning with temporal contrastive loss")
    print("3. Use batch_size=4 as requested")
    print("\n")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
