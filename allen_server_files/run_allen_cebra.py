#!/usr/bin/env python3
"""
Main script to train CEBRA on Allen Brain Observatory data
with poyo-ssl style self-supervised learning

Usage:
    python run_allen_cebra.py --config allen_config.yaml
    python run_allen_cebra.py --config allen_config.yaml --data_dir /path/to/data
"""

import argparse
import yaml
import h5py
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_allen_session(h5_path, split='train', normalize=True):
    """
    Load Allen Brain Observatory session data

    Args:
        h5_path: Path to h5 file
        split: 'train', 'valid', or 'test'
        normalize: Apply z-score normalization

    Returns:
        dict with neural_data, timestamps, metadata
    """
    print(f"   Loading {h5_path.name} ({split} split)...")

    with h5py.File(h5_path, 'r') as f:
        # Load calcium traces
        df_over_f = f['calcium_traces/df_over_f'][:]

        # Load mask
        mask = f[f'calcium_traces/{split}_mask'][:]

        # Apply mask
        neural_data = df_over_f[mask]

        # Normalize if requested
        if normalize:
            mean = neural_data.mean(axis=0, keepdims=True)
            std = neural_data.std(axis=0, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            neural_data = (neural_data - mean) / std

        # Get timestamps
        if 'calcium_traces/domain/start' in f and 'calcium_traces/domain/end' in f:
            domain_start = f['calcium_traces/domain/start'][0]
            domain_end = f['calcium_traces/domain/end'][0]
            all_timestamps = np.linspace(domain_start, domain_end, len(df_over_f))
            timestamps = all_timestamps[mask]
        else:
            timestamps = np.arange(len(neural_data)).astype(np.float32)

        # Normalize timestamps to [0, 1]
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

        metadata = {
            'session_id': h5_path.stem,
            'split': split,
            'num_timepoints': neural_data.shape[0],
            'num_neurons': neural_data.shape[1],
        }

        print(f"      Shape: {neural_data.shape}, Range: [{neural_data.min():.4f}, {neural_data.max():.4f}]")

        return {
            'neural_data': neural_data,
            'timestamps': timestamps,
            'metadata': metadata
        }

def load_dataset(config, split='train'):
    """
    Load complete dataset from multiple sessions

    Args:
        config: Configuration dictionary
        split: 'train', 'valid', or 'test'

    Returns:
        List of data dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Loading {split} dataset")
    print(f"{'='*80}")

    data_dir = Path(config['dataset']['data_dir'])
    h5_files = sorted(data_dir.glob(config['dataset']['file_pattern']))

    max_sessions = config['dataset']['sessions'].get('max_sessions', None)
    if max_sessions is not None:
        h5_files = h5_files[:max_sessions]

    print(f"Found {len(h5_files)} sessions")

    all_data = []
    for h5_file in h5_files:
        try:
            data = load_allen_session(
                h5_file,
                split=split,
                normalize=config['dataset']['neural'].get('normalize', True)
            )
            all_data.append(data)
        except Exception as e:
            print(f"   ⚠️  Error loading {h5_file.name}: {e}")

    print(f"\n✅ Loaded {len(all_data)} sessions")
    total_neurons = sum(d['metadata']['num_neurons'] for d in all_data)
    total_timepoints = sum(d['metadata']['num_timepoints'] for d in all_data)
    print(f"   Total neurons: {total_neurons}")
    print(f"   Total timepoints: {total_timepoints}")

    return all_data

def create_pytorch_dataset(data_list, mode='separate'):
    """
    Create PyTorch dataset from list of session data

    Args:
        data_list: List of data dictionaries
        mode: 'separate' - keep sessions separate (for multi-session CEBRA)
              'pad' - pad to max neurons and concatenate
              'first_only' - use only first session

    Returns:
        List of (neural_tensor, time_tensor) tuples for each session
    """
    if mode == 'separate':
        # Keep sessions separate for multi-session training
        datasets = []
        for data in data_list:
            neural_tensor = torch.from_numpy(data['neural_data']).float()
            time_tensor = torch.from_numpy(data['timestamps']).float()
            datasets.append((neural_tensor, time_tensor))
        return datasets

    elif mode == 'pad':
        # Pad all sessions to have same number of neurons
        max_neurons = max(d['neural_data'].shape[1] for d in data_list)

        padded_neural = []
        all_timestamps = []

        for data in data_list:
            neural = data['neural_data']
            n_neurons = neural.shape[1]

            if n_neurons < max_neurons:
                # Pad with zeros
                padding = np.zeros((neural.shape[0], max_neurons - n_neurons))
                neural = np.concatenate([neural, padding], axis=1)

            padded_neural.append(neural)
            all_timestamps.append(data['timestamps'])

        # Concatenate all sessions
        all_neural = np.concatenate(padded_neural, axis=0)
        all_timestamps = np.concatenate(all_timestamps, axis=0)

        neural_tensor = torch.from_numpy(all_neural).float()
        time_tensor = torch.from_numpy(all_timestamps).float()

        return [(neural_tensor, time_tensor)]

    elif mode == 'first_only':
        # Use only first session
        data = data_list[0]
        neural_tensor = torch.from_numpy(data['neural_data']).float()
        time_tensor = torch.from_numpy(data['timestamps']).float()
        return [(neural_tensor, time_tensor)]

    else:
        raise ValueError(f"Unknown mode: {mode}")

def train_model(config, train_data, valid_data=None, mode='separate'):
    """
    Train CEBRA model

    Args:
        config: Configuration dictionary
        train_data: Training data list
        valid_data: Validation data list (optional)
        mode: 'separate', 'pad', or 'first_only'

    Returns:
        Trained model
    """
    print(f"\n{'='*80}")
    print("Training CEBRA Model")
    print(f"{'='*80}")

    # Create PyTorch datasets
    print(f"\nPreparing training data (mode={mode})...")
    train_datasets = create_pytorch_dataset(train_data, mode=mode)

    print(f"   Number of training sessions: {len(train_datasets)}")
    for i, (neural, time) in enumerate(train_datasets):
        print(f"   Session {i+1}: neural={neural.shape}, time={time.shape}")

    if valid_data is not None:
        print(f"\nPreparing validation data (mode={mode})...")
        valid_datasets = create_pytorch_dataset(valid_data, mode=mode)
        print(f"   Number of validation sessions: {len(valid_datasets)}")
        for i, (neural, time) in enumerate(valid_datasets):
            print(f"   Session {i+1}: neural={neural.shape}, time={time.shape}")

    # Create simple model placeholder
    # Note: Full CEBRA integration would require importing CEBRA properly
    print("\n" + "="*80)
    print("Model Training Configuration")
    print("="*80)
    print(f"   Architecture: {config['model']['architecture']}")
    print(f"   Batch size: {config['model']['batch_size']}")
    print(f"   Learning rate: {config['model']['learning_rate']}")
    print(f"   Max iterations: {config['model']['max_iterations']}")
    print(f"   Output dimension: {config['model']['output_dimension']}")

    print("\n⚠️  Note: This is a data loading test.")
    print("   For full CEBRA training, ensure all dependencies are installed:")
    print("   - literate_dataclasses")
    print("   - cebra package with all dependencies")

    # Mock training loop for demonstration
    print("\n" + "="*80)
    print("Mock Training Loop")
    print("="*80)

    batch_size = config['model']['batch_size']

    # Process each session
    for session_idx, (train_neural, train_time) in enumerate(train_datasets):
        num_batches = len(train_neural) // batch_size

        print(f"\n   Session {session_idx + 1}:")
        print(f"      Total samples: {len(train_neural)}")
        print(f"      Batch size: {batch_size}")
        print(f"      Number of batches per epoch: {num_batches}")

        # Simulate a few batches
        print(f"      Simulating first 3 batches:")
        for i in range(min(3, num_batches)):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_neural = train_neural[start_idx:end_idx]
            batch_time = train_time[start_idx:end_idx]

            print(f"         Batch {i+1}: neural={batch_neural.shape}, time={batch_time.shape}")

    return {
        'train_datasets': train_datasets,
        'valid_datasets': valid_datasets if valid_data is not None else None,
        'config': config
    }

def save_results(model_dict, config):
    """
    Save model and results

    Args:
        model_dict: Dictionary containing model and data
        config: Configuration dictionary
    """
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")

    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save configuration
    config_path = output_dir / f"config_{timestamp}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"   ✅ Saved config to {config_path}")

    # Save data info
    train_shapes = [(neural.shape[0], neural.shape[1]) for neural, _ in model_dict['train_datasets']]

    info = {
        'timestamp': timestamp,
        'num_sessions': len(model_dict['train_datasets']),
        'train_shapes': train_shapes,
        'config': config,
    }

    info_path = output_dir / f"info_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"   ✅ Saved info to {info_path}")

    print(f"\n   All results saved to {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train CEBRA on Allen Brain Observatory data"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='allen_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Override data directory from config'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--max_sessions',
        type=int,
        default=None,
        help='Override max sessions from config'
    )

    args = parser.parse_args()

    # Load configuration
    print("\n" + "="*80)
    print("Allen Brain Observatory - CEBRA Training")
    print("="*80)
    print(f"\nLoading config from: {args.config}")

    config = load_config(args.config)

    # Override with command line arguments
    if args.data_dir is not None:
        config['dataset']['data_dir'] = args.data_dir
    if args.batch_size is not None:
        config['model']['batch_size'] = args.batch_size
        config['dataloader']['batch_size'] = args.batch_size
    if args.max_sessions is not None:
        config['dataset']['sessions']['max_sessions'] = args.max_sessions

    # Set seed for reproducibility
    if config.get('seed') is not None:
        print(f"Setting seed to {config['seed']}")
        set_seed(config['seed'])

    # Load datasets
    train_data = load_dataset(config, split='train')
    valid_data = load_dataset(config, split='valid')

    # Train model
    model_dict = train_model(config, train_data, valid_data)

    # Save results
    save_results(model_dict, config)

    print("\n" + "="*80)
    print("✅ All Done!")
    print("="*80)
    print("\nNext steps:")
    print("1. Install full CEBRA dependencies if needed")
    print("2. Integrate with actual CEBRA model training")
    print("3. Run on GPU for faster training")
    print(f"4. Results saved to {config['output']['save_dir']}/")
    print()

if __name__ == "__main__":
    main()
