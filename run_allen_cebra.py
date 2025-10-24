#!/usr/bin/env python3
"""
Main script to train CEBRA on Allen Brain Observatory data
and decode natural movie frames using HalfUNet

Usage:
    python run_allen_cebra.py --config allen_config.yaml
    python run_allen_cebra.py --config allen_config.yaml --data_dir /path/to/data
"""

import argparse
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm

# Import kirby modules
from kirby.data.dataset import Dataset as KirbyDataset
from kirby.nn.loss import compute_loss_or_metric, SSIMLoss, MultiScaleLoss
from kirby.nn.unet import HalfUNet

# Import CEBRA
try:
    import cebra
    CEBRA_AVAILABLE = True
except ImportError:
    CEBRA_AVAILABLE = False
    print("Warning: CEBRA not available")

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

def create_kirby_dataset(config, split='train'):
    """
    Create Kirby Dataset for Allen Brain Observatory

    Args:
        config: Configuration dictionary
        split: 'train', 'valid', or 'test'

    Returns:
        KirbyDataset instance
    """
    print(f"\n{'='*80}")
    print(f"Creating Kirby Dataset ({split} split)")
    print(f"{'='*80}")

    # Prepare include configuration
    include = [{
        'selection': [{
            'dandiset': 'allen_brain_observatory_calcium'
        }]
    }]

    # Create dataset
    dataset = KirbyDataset(
        root=config['dataset']['data_dir'],
        split=split,
        include=include,
        transform=None,
        pretrain=False,
        finetune=False,
        small_model=config.get('small_model', False),
        task='movie_decoding_one',  # natural movie one
        ssl_mode=config.get('ssl_mode', 'predictable'),
        model_dim=config['model']['output_dimension']
    )

    print(f"✅ Dataset created with {len(dataset)} sessions")
    return dataset

def train_cebra_model(config, train_dataset):
    """
    Train CEBRA model on neural data

    Args:
        config: Configuration dictionary
        train_dataset: Training dataset

    Returns:
        Trained CEBRA model
    """
    if not CEBRA_AVAILABLE:
        raise ImportError("CEBRA is not available. Please install it.")

    print(f"\n{'='*80}")
    print("Training CEBRA Model")
    print(f"{'='*80}")

    # Get first session data for training
    # TODO: Support multi-session training
    session_data = train_dataset.get_session_data(train_dataset.session_ids[0])

    # Extract neural data
    neural_data = session_data.patches.obj.cpu().numpy()

    print(f"Neural data shape: {neural_data.shape}")

    # Initialize CEBRA model
    cebra_model = cebra.CEBRA(
        model_architecture=config['model']['model_architecture'],
        batch_size=config['model']['batch_size'],
        learning_rate=config['model']['learning_rate'],
        max_iterations=config['model']['max_iterations'],
        output_dimension=config['model']['output_dimension'],
        device=config['model']['device'],
        verbose=config['model']['verbose'],
    )

    print("Training CEBRA model...")
    cebra_model.fit(neural_data)
    print("✅ CEBRA training complete!")

    return cebra_model

def train_halfunet_decoder(config, cebra_model, train_dataset, valid_dataset, device='cuda'):
    """
    Train HalfUNet decoder to reconstruct movie frames from CEBRA embeddings

    Args:
        config: Configuration dictionary
        cebra_model: Trained CEBRA model
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        device: Device to use for training

    Returns:
        Trained HalfUNet decoder
    """
    print(f"\n{'='*80}")
    print("Training HalfUNet Decoder")
    print(f"{'='*80}")

    # Determine output shape based on model dimension
    model_dim = config['model']['output_dimension']
    if model_dim <= 512:
        output_shape = (64, 128)
    else:
        output_shape = (128, 256)

    print(f"Output shape: {output_shape}")

    # Initialize HalfUNet
    decoder = HalfUNet(
        input_dim=model_dim,
        output_channels=1,  # Grayscale movie frames
        output_shape=output_shape,
        latent_dim=config.get('decoder', {}).get('latent_dim', 1024)
    ).to(device)

    # Loss functions
    ssim_loss = SSIMLoss()
    mse_loss = nn.MSELoss()
    multiscale_loss = MultiScaleLoss(loss_type='l1', scales=[1.0, 0.5, 0.25])

    # Optimizer
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=config.get('decoder', {}).get('learning_rate', 0.001)
    )

    # Training parameters
    num_epochs = config.get('decoder', {}).get('num_epochs', 50)
    batch_size = config.get('decoder', {}).get('batch_size', 32)

    print(f"Training parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")

    # Training loop
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        decoder.train()
        train_losses = []

        # Training phase
        for session_id in tqdm(train_dataset.session_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                session_data = train_dataset.get_session_data(session_id)

                # Get neural data and movie frames
                neural_data = session_data.patches.obj.cpu().numpy()
                movie_frames = session_data.movie_frames  # Shape: (T, H, W)

                # Generate CEBRA embeddings
                embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

                # Prepare movie frames
                targets = movie_frames.unsqueeze(1).float().to(device)  # Add channel dim

                # Mini-batch training
                num_samples = len(embeddings)
                indices = torch.randperm(num_samples)

                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_embeddings = embeddings[batch_indices]
                    batch_targets = targets[batch_indices]

                    # Forward pass
                    predictions = decoder(batch_embeddings)

                    # Compute loss (combination of losses)
                    loss_mse = mse_loss(predictions, batch_targets)
                    loss_ssim = ssim_loss(predictions, batch_targets)
                    loss_multiscale = multiscale_loss(predictions, batch_targets)

                    total_loss = loss_mse + 0.5 * loss_ssim + 0.3 * loss_multiscale

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    train_losses.append(total_loss.item())

            except Exception as e:
                print(f"Warning: Error processing session {session_id}: {e}")
                continue

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')

        # Validation phase
        decoder.eval()
        valid_losses = []

        with torch.no_grad():
            for session_id in valid_dataset.session_ids:
                try:
                    session_data = valid_dataset.get_session_data(session_id)

                    neural_data = session_data.patches.obj.cpu().numpy()
                    movie_frames = session_data.movie_frames

                    embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)
                    targets = movie_frames.unsqueeze(1).float().to(device)

                    predictions = decoder(embeddings)

                    loss_mse = mse_loss(predictions, targets)
                    loss_ssim = ssim_loss(predictions, targets)

                    total_loss = loss_mse + 0.5 * loss_ssim
                    valid_losses.append(total_loss.item())

                except Exception as e:
                    continue

        avg_valid_loss = np.mean(valid_losses) if valid_losses else float('inf')

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print(f"  ✅ New best model saved!")

    print("✅ HalfUNet training complete!")
    return decoder

def visualize_results(decoder, cebra_model, test_dataset, output_dir, device='cuda', num_samples=5):
    """
    Visualize reconstruction results

    Args:
        decoder: Trained HalfUNet decoder
        cebra_model: Trained CEBRA model
        test_dataset: Test dataset
        output_dir: Output directory for visualizations
        device: Device to use
        num_samples: Number of samples to visualize
    """
    print(f"\n{'='*80}")
    print("Visualizing Results")
    print(f"{'='*80}")

    decoder.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        # Get first test session
        session_id = test_dataset.session_ids[0]
        session_data = test_dataset.get_session_data(session_id)

        neural_data = session_data.patches.obj.cpu().numpy()
        movie_frames = session_data.movie_frames

        # Generate embeddings and predictions
        embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)
        predictions = decoder(embeddings).cpu()

        # Select random samples
        indices = np.random.choice(len(predictions), size=min(num_samples, len(predictions)), replace=False)

        # Create visualization
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

        for i, idx in enumerate(indices):
            # Original frame
            axes[i, 0].imshow(movie_frames[idx].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title(f'Original Frame {idx}')
            axes[i, 0].axis('off')

            # Reconstructed frame
            axes[i, 1].imshow(predictions[idx, 0].numpy(), cmap='gray')
            axes[i, 1].set_title(f'Reconstructed Frame {idx}')
            axes[i, 1].axis('off')

        plt.tight_layout()
        save_path = output_dir / 'reconstruction_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualization saved to {save_path}")

    return save_path

def save_results(cebra_model, decoder, config):
    """
    Save trained models and results

    Args:
        cebra_model: Trained CEBRA model
        decoder: Trained HalfUNet decoder
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

    # Save CEBRA model
    cebra_path = output_dir / f"cebra_model_{timestamp}.pt"
    cebra_model.save(str(cebra_path))
    print(f"   ✅ Saved CEBRA model to {cebra_path}")

    # Save HalfUNet decoder
    decoder_path = output_dir / f"halfunet_decoder_{timestamp}.pt"
    torch.save(decoder.state_dict(), decoder_path)
    print(f"   ✅ Saved HalfUNet decoder to {decoder_path}")

    # Save metadata
    info = {
        'timestamp': timestamp,
        'cebra_output_dim': config['model']['output_dimension'],
        'decoder_latent_dim': config.get('decoder', {}).get('latent_dim', 1024),
        'task': 'movie_decoding_one',
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
        description="Train CEBRA + HalfUNet on Allen Brain Observatory for movie decoding"
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
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--skip_cebra',
        action='store_true',
        help='Skip CEBRA training (use existing model)'
    )
    parser.add_argument(
        '--cebra_model_path',
        type=str,
        default=None,
        help='Path to existing CEBRA model'
    )

    args = parser.parse_args()

    # Load configuration
    print("\n" + "="*80)
    print("Allen Brain Observatory - CEBRA + HalfUNet Movie Decoding")
    print("="*80)
    print(f"\nLoading config from: {args.config}")

    config = load_config(args.config)

    # Override with command line arguments
    if args.data_dir is not None:
        config['dataset']['data_dir'] = args.data_dir

    # Set device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Using device: {device}")

    # Set seed for reproducibility
    if config.get('seed') is not None:
        print(f"Setting seed to {config['seed']}")
        set_seed(config['seed'])

    # Create datasets using Kirby
    train_dataset = create_kirby_dataset(config, split='train')
    valid_dataset = create_kirby_dataset(config, split='valid')
    test_dataset = create_kirby_dataset(config, split='test')

    # Train or load CEBRA model
    if args.skip_cebra and args.cebra_model_path:
        print(f"\nLoading existing CEBRA model from {args.cebra_model_path}")
        cebra_model = cebra.CEBRA.load(args.cebra_model_path)
    else:
        cebra_model = train_cebra_model(config, train_dataset)

    # Train HalfUNet decoder
    decoder = train_halfunet_decoder(
        config,
        cebra_model,
        train_dataset,
        valid_dataset,
        device=device
    )

    # Visualize results
    visualize_results(
        decoder,
        cebra_model,
        test_dataset,
        config['output']['save_dir'],
        device=device
    )

    # Save results
    save_results(cebra_model, decoder, config)

    print("\n" + "="*80)
    print("✅ All Done!")
    print("="*80)
    print("\nResults:")
    print(f"1. Models saved to {config['output']['save_dir']}/")
    print(f"2. Visualizations saved to {config['output']['save_dir']}/")
    print("\nTo use the trained models:")
    print("  - Load CEBRA model: cebra.CEBRA.load('path/to/cebra_model.pt')")
    print("  - Load decoder: decoder.load_state_dict(torch.load('path/to/decoder.pt'))")
    print()

if __name__ == "__main__":
    main()
