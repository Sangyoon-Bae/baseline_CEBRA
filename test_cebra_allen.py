#!/usr/bin/env python3
"""
Test script to load Allen Brain Observatory data with CEBRA
Using poyo-ssl style (self-supervised learning) with batch_size=4
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# CEBRA imports
import cebra
from cebra.data import TensorDataset, DatasetCollection
from cebra.data import ContinuousDataLoader

def load_allen_session_as_cebra_dataset(h5_path, split='train'):
    """
    Load Allen Brain Observatory session as CEBRA TensorDataset

    Args:
        h5_path: Path to h5 file
        split: 'train', 'valid', or 'test'

    Returns:
        TensorDataset with neural data and continuous index
    """
    print(f"\n{'='*80}")
    print(f"Loading {h5_path.name} - {split} split")
    print(f"{'='*80}")

    with h5py.File(h5_path, 'r') as f:
        # Load calcium traces
        df_over_f = f['calcium_traces/df_over_f'][:]  # (timepoints, neurons)

        # Load mask for the split
        mask = f[f'calcium_traces/{split}_mask'][:]

        # Apply mask
        neural_data = df_over_f[mask]

        print(f"   Neural data shape: {neural_data.shape}")
        print(f"   Data type: {neural_data.dtype}")
        print(f"   Data range: [{neural_data.min():.4f}, {neural_data.max():.4f}]")

        # Create continuous index (time)
        # For self-supervised learning, we use timestamps as continuous variable
        if 'calcium_traces/domain' in f:
            domain_start = f['calcium_traces/domain/start'][0]
            domain_end = f['calcium_traces/domain/end'][0]

            # Create timestamps for masked data
            all_timestamps = np.linspace(domain_start, domain_end, len(df_over_f))
            timestamps = all_timestamps[mask]

            # Normalize to [0, 1] for CEBRA
            timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
            continuous_index = timestamps_norm.reshape(-1, 1)
        else:
            # Fallback: use frame indices
            continuous_index = np.arange(len(neural_data)).reshape(-1, 1).astype(np.float32)
            continuous_index = continuous_index / continuous_index.max()

        print(f"   Continuous index shape: {continuous_index.shape}")
        print(f"   Continuous index range: [{continuous_index.min():.4f}, {continuous_index.max():.4f}]")

        # Convert to tensors
        neural_tensor = torch.from_numpy(neural_data).float()
        continuous_tensor = torch.from_numpy(continuous_index).float()

        # Create CEBRA TensorDataset
        dataset = TensorDataset(
            neural=neural_tensor,
            continuous=continuous_tensor
        )

        print(f"   ✅ Created TensorDataset with {len(dataset)} samples")
        print(f"   Input dimension: {dataset.input_dimension}")

        return dataset

def test_dataloader(dataset, batch_size=4):
    """Test CEBRA dataloader with given batch size"""
    print(f"\n{'='*80}")
    print(f"Testing DataLoader with batch_size={batch_size}")
    print(f"{'='*80}")

    try:
        # Create continuous dataloader (for self-supervised learning)
        dataloader = ContinuousDataLoader(
            dataset=dataset,
            num_steps=10,
            batch_size=batch_size,
            time_offset=10,
        )

        print(f"   ✅ Created ContinuousDataLoader")
        print(f"   Batch size: {batch_size}")
        print(f"   Num steps: 10")
        print(f"   Time offset: 10")

        # Test iteration
        print(f"\n   Testing iteration...")
        batch = next(iter(dataloader))

        print(f"\n   Batch structure:")
        print(f"      - reference: {batch.reference.shape}")
        print(f"      - positive: {batch.positive.shape}")
        print(f"      - negative: {batch.negative.shape}")

        print(f"\n   ✅ DataLoader test successful!")

        return dataloader

    except Exception as e:
        print(f"   ❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_multisession_dataset(data_dir="data", split='train', max_sessions=4):
    """Create a multi-session dataset from multiple h5 files"""
    print(f"\n{'='*80}")
    print(f"Creating Multi-Session Dataset")
    print(f"{'='*80}")

    h5_files = sorted(Path(data_dir).glob("*.h5"))[:max_sessions]

    print(f"   Found {len(h5_files)} h5 files (using {max_sessions})")

    datasets = []
    for h5_file in h5_files:
        try:
            dataset = load_allen_session_as_cebra_dataset(h5_file, split=split)
            datasets.append(dataset)
        except Exception as e:
            print(f"   ⚠️  Skipping {h5_file.name}: {e}")

    if len(datasets) == 0:
        print(f"   ❌ No datasets loaded!")
        return None

    # Create DatasetCollection
    print(f"\n   Creating DatasetCollection with {len(datasets)} sessions...")
    collection = DatasetCollection(*datasets)

    print(f"   ✅ Created DatasetCollection")
    print(f"   Number of sessions: {collection.num_sessions}")
    print(f"   Input dimensions: {[collection.get_input_dimension(i) for i in range(min(3, collection.num_sessions))]}")

    return collection

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("Allen Brain Observatory - CEBRA Test (poyo-ssl style)")
    print("="*80)

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        return 1

    # Test 1: Single session
    print("\n" + "="*80)
    print("TEST 1: Single Session Dataset")
    print("="*80)

    h5_files = sorted(data_dir.glob("*.h5"))
    if len(h5_files) == 0:
        print(f"❌ No h5 files found!")
        return 1

    test_file = h5_files[0]
    dataset = load_allen_session_as_cebra_dataset(test_file, split='train')

    if dataset is not None:
        dataloader = test_dataloader(dataset, batch_size=4)

    # Test 2: Multi-session
    print("\n" + "="*80)
    print("TEST 2: Multi-Session Dataset")
    print("="*80)

    collection = create_multisession_dataset(data_dir, split='train', max_sessions=4)

    if collection is not None:
        # Test dataloader with multi-session dataset
        # Note: For multi-session, we need a multi-session dataloader
        print(f"\n   Multi-session dataset created successfully!")
        print(f"   To train CEBRA on this dataset, you can use:")
        print(f"   ```python")
        print(f"   cebra_model = cebra.CEBRA(")
        print(f"       model_architecture='offset10-model',")
        print(f"       batch_size=4,")
        print(f"       learning_rate=3e-4,")
        print(f"       temperature=1,")
        print(f"       output_dimension=32,")
        print(f"       max_iterations=5000,")
        print(f"       distance='cosine',")
        print(f"       conditional='time_delta',")
        print(f"       device='cuda_if_available',")
        print(f"       verbose=True,")
        print(f"   )")
        print(f"   cebra_model.fit(collection)")
        print(f"   ```")

    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
