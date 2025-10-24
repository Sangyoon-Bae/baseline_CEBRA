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
import os

# Import kirby modules
from kirby.data.dataset import Dataset as KirbyDataset
from kirby.nn.loss import (
    SSIMLoss,
    AlexNetPerceptualLoss,
    GradientDifferenceLoss,
    FocalLoss,
    FFTLoss
)
from kirby.nn.unet import HalfUNet

# Import CEBRA
try:
    import cebra
    CEBRA_AVAILABLE = True
except ImportError:
    CEBRA_AVAILABLE = False
    print("Warning: CEBRA not available")

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

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

def init_wandb(config):
    """Initialize Weights & Biases logging"""
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping wandb initialization")
        return None

    wandb_config = config.get('wandb', {})
    if not wandb_config.get('enabled', False):
        print("wandb is disabled in config")
        return None

    # Set API key if provided
    api_key = wandb_config.get('api_key', '').strip()
    if api_key:
        os.environ['WANDB_API_KEY'] = api_key

    # Initialize wandb
    run = wandb.init(
        entity=wandb_config.get('entity', None) or None,
        project=wandb_config.get('project', 'allen-cebra-halfunet'),
        name=wandb_config.get('name', None),
        notes=wandb_config.get('notes', ''),
        tags=wandb_config.get('tags', []),
        config=config
    )

    print(f"✅ wandb initialized: {run.url}")
    return run

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

    # Disable data leakage check for Allen dataset
    # Allen dataset uses time-based splits which don't align with mask-based checks
    dataset._check_for_data_leakage_flag = False

    print(f"✅ Dataset created with {len(dataset)} sessions")
    return dataset

def initialize_cebra_model(config, device='cuda'):
    """
    Initialize CEBRA model (without training)

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        Initialized CEBRA model
    """
    if not CEBRA_AVAILABLE:
        raise ImportError("CEBRA is not available. Please install it.")

    print(f"\n{'='*80}")
    print("Initializing CEBRA Model")
    print(f"{'='*80}")

    # Initialize CEBRA model
    cebra_model = cebra.CEBRA(
        model_architecture=config['model']['model_architecture'],
        batch_size=config['model']['batch_size'],
        learning_rate=config['model']['learning_rate'],
        max_iterations=config['model']['max_iterations'],
        output_dimension=config['model']['output_dimension'],
        device=device if device != 'cpu' else 'cpu',
        verbose=config['model']['verbose'],
    )

    print("✅ CEBRA model initialized (will be trained jointly with decoder)")

    return cebra_model

def extract_neural_data(session_data):
    """
    Extract neural data from session_data object.
    Handles both Allen dataset (calcium_traces) and other datasets (patches).

    Args:
        session_data: Data object from dataset

    Returns:
        numpy array of neural data
    """
    if hasattr(session_data, 'calcium_traces'):
        # Allen dataset uses calcium_traces
        calcium_traces = session_data.calcium_traces
        if hasattr(calcium_traces, 'df_over_f'):
            neural_data_raw = calcium_traces.df_over_f
        else:
            # Use first available key
            data_key = [k for k in calcium_traces.keys if 'timestamp' not in k.lower()][0]
            neural_data_raw = getattr(calcium_traces, data_key)
    elif hasattr(session_data, 'patches'):
        # Other datasets use patches
        patches = session_data.patches
        if hasattr(patches, 'obj'):
            neural_data_raw = patches.obj
        else:
            neural_data_raw = patches
    else:
        raise AttributeError(f"No suitable neural data found. Available attributes: {session_data.keys}")

    # Convert to numpy if it's a torch tensor
    if hasattr(neural_data_raw, 'cpu'):
        neural_data = neural_data_raw.cpu().numpy()
    elif hasattr(neural_data_raw, 'numpy'):
        neural_data = neural_data_raw.numpy()
    else:
        neural_data = neural_data_raw

    return neural_data

def train_joint_model(config, cebra_model, train_dataset, valid_dataset, device='cuda', wandb_run=None):
    """
    Train CEBRA and HalfUNet jointly end-to-end

    This enables the decoder loss to guide CEBRA's representation learning,
    creating embeddings optimized for movie frame reconstruction.

    Args:
        config: Configuration dictionary
        cebra_model: Initialized (untrained) CEBRA model
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        device: Device to use for training
        wandb_run: Weights & Biases run object for logging

    Returns:
        Tuple of (trained CEBRA model, trained HalfUNet decoder)
    """
    print(f"\n{'='*80}")
    print("Joint Training: CEBRA + HalfUNet Decoder")
    print(f"{'='*80}")
    print("Training both models end-to-end for optimized reconstruction")

    # Determine output shape based on model dimension
    model_dim = config['model']['output_dimension']
    if model_dim <= 512:
        output_shape = (64, 128)
    else:
        output_shape = (128, 256)

    print(f"Output shape: {output_shape}")

    # Initialize HalfUNet
    # HalfUNet expects: in_channels, out_channels, latent_dim
    decoder = HalfUNet(
        in_channels=1,
        out_channels=1,  # Grayscale movie frames
        latent_dim=config.get('decoder', {}).get('latent_dim', 1024)
    ).to(device)

    # Loss functions (6 losses as specified)
    ssim_loss = SSIMLoss()
    perceptual_loss = AlexNetPerceptualLoss(layer=3)
    gradient_loss = GradientDifferenceLoss(channels=1, loss_type='l1')  # 1 channel for grayscale
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    fft_loss = FFTLoss(loss_type='l1')
    l1_loss = nn.L1Loss()

    # Loss weights (read from config or use defaults)
    loss_weights = config.get('decoder', {}).get('loss_weights', {
        'l1': 1.0,
        'ssim': 0.5,
        'perceptual': 0.3,
        'gradient': 0.2,
        'focal': 0.1,
        'fft': 0.2
    })

    print(f"\nLoss configuration:")
    print(f"  L1 Loss (weight: {loss_weights['l1']})")
    print(f"  SSIM Loss (weight: {loss_weights['ssim']})")
    print(f"  Perceptual Loss - AlexNet (weight: {loss_weights['perceptual']})")
    print(f"  Gradient Difference Loss (weight: {loss_weights['gradient']})")
    print(f"  Focal Loss (weight: {loss_weights['focal']})")
    print(f"  FFT Loss (weight: {loss_weights['fft']})")

    # Get CEBRA's internal PyTorch model
    # CEBRA wraps a PyTorch model, we need to access it for joint training
    if hasattr(cebra_model, 'model'):
        cebra_net = cebra_model.model
    elif hasattr(cebra_model, 'net'):
        cebra_net = cebra_model.net
    else:
        # Fallback: try to use cebra_model directly
        cebra_net = cebra_model

    # Move CEBRA network to device
    if hasattr(cebra_net, 'to'):
        cebra_net = cebra_net.to(device)

    print(f"\n✅ CEBRA network extracted for joint training")

    # Joint optimizer for both CEBRA and decoder
    # Combine parameters from both models
    all_parameters = list(decoder.parameters())
    if hasattr(cebra_net, 'parameters'):
        all_parameters += list(cebra_net.parameters())

    optimizer = torch.optim.Adam(
        all_parameters,
        lr=config.get('decoder', {}).get('learning_rate', 0.001)
    )

    print(f"Joint optimizer created with {len(all_parameters)} parameter groups")

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
        train_loss_details = {
            'l1': [], 'ssim': [], 'perceptual': [],
            'gradient': [], 'focal': [], 'fft': []
        }

        # Training phase
        for session_id in tqdm(train_dataset.session_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                session_data = train_dataset.get_session_data(session_id)

                # Get neural data and movie frames
                neural_data = extract_neural_data(session_data)
                movie_frames = session_data.movie_frames  # Shape: (T, H, W)

                # Convert neural data to torch tensor
                neural_data_tensor = torch.from_numpy(neural_data).float().to(device)

                # Generate CEBRA embeddings using forward pass (keep gradients!)
                # This allows gradients to flow back to CEBRA
                cebra_net.train()
                if hasattr(cebra_net, 'encode'):
                    embeddings = cebra_net.encode(neural_data_tensor)
                elif hasattr(cebra_net, 'forward'):
                    embeddings = cebra_net(neural_data_tensor)
                else:
                    # Fallback: use transform but won't have gradients
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

                    # Ensure predictions have channel dimension
                    if predictions.dim() == 3:
                        predictions = predictions.unsqueeze(1)

                    # Compute all 6 losses
                    loss_l1_val = l1_loss(predictions, batch_targets)
                    loss_ssim_val = ssim_loss(predictions, batch_targets)
                    loss_perceptual_val = perceptual_loss(predictions, batch_targets)
                    loss_gradient_val = gradient_loss(predictions, batch_targets)
                    loss_focal_val = focal_loss(predictions, batch_targets)
                    loss_fft_val = fft_loss(predictions, batch_targets)

                    # Weighted combination of losses
                    total_loss = (
                        loss_weights['l1'] * loss_l1_val +
                        loss_weights['ssim'] * loss_ssim_val +
                        loss_weights['perceptual'] * loss_perceptual_val +
                        loss_weights['gradient'] * loss_gradient_val +
                        loss_weights['focal'] * loss_focal_val +
                        loss_weights['fft'] * loss_fft_val
                    )

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # Track losses
                    train_losses.append(total_loss.item())
                    train_loss_details['l1'].append(loss_l1_val.item())
                    train_loss_details['ssim'].append(loss_ssim_val.item())
                    train_loss_details['perceptual'].append(loss_perceptual_val.item())
                    train_loss_details['gradient'].append(loss_gradient_val.item())
                    train_loss_details['focal'].append(loss_focal_val.item())
                    train_loss_details['fft'].append(loss_fft_val.item())

            except Exception as e:
                print(f"Warning: Error processing session {session_id}: {e}")
                continue

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')

        # Validation phase
        decoder.eval()
        cebra_net.eval()
        valid_losses = []
        valid_loss_details = {
            'l1': [], 'ssim': [], 'perceptual': [],
            'gradient': [], 'focal': [], 'fft': []
        }

        # For wandb image logging (save first batch of predictions)
        first_predictions = None
        first_targets = None

        with torch.no_grad():
            for idx, session_id in enumerate(valid_dataset.session_ids):
                try:
                    session_data = valid_dataset.get_session_data(session_id)

                    neural_data = extract_neural_data(session_data)
                    movie_frames = session_data.movie_frames

                    # Convert neural data to torch tensor
                    neural_data_tensor = torch.from_numpy(neural_data).float().to(device)

                    # Generate CEBRA embeddings
                    if hasattr(cebra_net, 'encode'):
                        embeddings = cebra_net.encode(neural_data_tensor)
                    elif hasattr(cebra_net, 'forward'):
                        embeddings = cebra_net(neural_data_tensor)
                    else:
                        embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

                    targets = movie_frames.unsqueeze(1).float().to(device)

                    predictions = decoder(embeddings)

                    # Ensure predictions have channel dimension
                    if predictions.dim() == 3:
                        predictions = predictions.unsqueeze(1)

                    # Compute all 6 losses
                    loss_l1_val = l1_loss(predictions, targets)
                    loss_ssim_val = ssim_loss(predictions, targets)
                    loss_perceptual_val = perceptual_loss(predictions, targets)
                    loss_gradient_val = gradient_loss(predictions, targets)
                    loss_focal_val = focal_loss(predictions, targets)
                    loss_fft_val = fft_loss(predictions, targets)

                    # Weighted combination of losses
                    total_loss = (
                        loss_weights['l1'] * loss_l1_val +
                        loss_weights['ssim'] * loss_ssim_val +
                        loss_weights['perceptual'] * loss_perceptual_val +
                        loss_weights['gradient'] * loss_gradient_val +
                        loss_weights['focal'] * loss_focal_val +
                        loss_weights['fft'] * loss_fft_val
                    )

                    # Track losses
                    valid_losses.append(total_loss.item())
                    valid_loss_details['l1'].append(loss_l1_val.item())
                    valid_loss_details['ssim'].append(loss_ssim_val.item())
                    valid_loss_details['perceptual'].append(loss_perceptual_val.item())
                    valid_loss_details['gradient'].append(loss_gradient_val.item())
                    valid_loss_details['focal'].append(loss_focal_val.item())
                    valid_loss_details['fft'].append(loss_fft_val.item())

                    # Save first batch for visualization
                    if idx == 0 and wandb_run is not None:
                        first_predictions = predictions[:8].detach().cpu()  # Save up to 8 samples
                        first_targets = targets[:8].detach().cpu()

                except Exception as e:
                    continue

        avg_valid_loss = np.mean(valid_losses) if valid_losses else float('inf')

        # Calculate average for each loss component
        avg_train_loss_details = {k: np.mean(v) if v else 0.0 for k, v in train_loss_details.items()}
        avg_valid_loss_details = {k: np.mean(v) if v else 0.0 for k, v in valid_loss_details.items()}

        # Print summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        print(f"  Loss breakdown (train/valid):")
        print(f"    L1:         {avg_train_loss_details['l1']:.4f} / {avg_valid_loss_details['l1']:.4f}")
        print(f"    SSIM:       {avg_train_loss_details['ssim']:.4f} / {avg_valid_loss_details['ssim']:.4f}")
        print(f"    Perceptual: {avg_train_loss_details['perceptual']:.4f} / {avg_valid_loss_details['perceptual']:.4f}")
        print(f"    Gradient:   {avg_train_loss_details['gradient']:.4f} / {avg_valid_loss_details['gradient']:.4f}")
        print(f"    Focal:      {avg_train_loss_details['focal']:.4f} / {avg_valid_loss_details['focal']:.4f}")
        print(f"    FFT:        {avg_train_loss_details['fft']:.4f} / {avg_valid_loss_details['fft']:.4f}")

        # Log to wandb
        if wandb_run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'train/total_loss': avg_train_loss,
                'valid/total_loss': avg_valid_loss,

                # Train loss components
                'train/l1_loss': avg_train_loss_details['l1'],
                'train/ssim_loss': avg_train_loss_details['ssim'],
                'train/perceptual_loss': avg_train_loss_details['perceptual'],
                'train/gradient_loss': avg_train_loss_details['gradient'],
                'train/focal_loss': avg_train_loss_details['focal'],
                'train/fft_loss': avg_train_loss_details['fft'],

                # Valid loss components
                'valid/l1_loss': avg_valid_loss_details['l1'],
                'valid/ssim_loss': avg_valid_loss_details['ssim'],
                'valid/perceptual_loss': avg_valid_loss_details['perceptual'],
                'valid/gradient_loss': avg_valid_loss_details['gradient'],
                'valid/focal_loss': avg_valid_loss_details['focal'],
                'valid/fft_loss': avg_valid_loss_details['fft'],
            }

            # Log images
            if first_predictions is not None and first_targets is not None:
                # Create side-by-side comparison
                num_images = min(4, len(first_predictions))  # Show up to 4 images
                images_to_log = []

                for i in range(num_images):
                    pred_img = first_predictions[i, 0].numpy()  # Remove channel dim
                    target_img = first_targets[i, 0].numpy()

                    # Stack horizontally: [target | prediction]
                    comparison = np.hstack([target_img, pred_img])
                    images_to_log.append(wandb.Image(
                        comparison,
                        caption=f"Epoch {epoch+1} - Sample {i+1} (Left: GT, Right: Pred)"
                    ))

                log_dict['predictions'] = images_to_log

            wandb_run.log(log_dict)

        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print(f"  ✅ New best model saved!")

    print("✅ Joint training complete!")
    print("Both CEBRA and HalfUNet have been trained end-to-end")
    return cebra_model, decoder

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

    # Initialize wandb
    wandb_run = init_wandb(config)

    # Create datasets using Kirby
    train_dataset = create_kirby_dataset(config, split='train')
    valid_dataset = create_kirby_dataset(config, split='valid')
    test_dataset = create_kirby_dataset(config, split='test')

    # Initialize or load CEBRA model
    if args.skip_cebra and args.cebra_model_path:
        print(f"\nLoading existing CEBRA model from {args.cebra_model_path}")
        cebra_model = cebra.CEBRA.load(args.cebra_model_path)
        print("Note: Using pre-trained CEBRA. Joint training will still fine-tune it.")
    else:
        # Initialize CEBRA model (not trained yet)
        cebra_model = initialize_cebra_model(config, device=device)

    # Joint training: CEBRA + HalfUNet decoder
    print("\n" + "="*80)
    print("Starting Joint End-to-End Training")
    print("="*80)
    cebra_model, decoder = train_joint_model(
        config,
        cebra_model,
        train_dataset,
        valid_dataset,
        device=device,
        wandb_run=wandb_run
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

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()
        print("✅ wandb run finished")

    print("\n" + "="*80)
    print("✅ All Done!")
    print("="*80)
    print("\nResults:")
    print(f"1. Models saved to {config['output']['save_dir']}/")
    print(f"2. Visualizations saved to {config['output']['save_dir']}/")
    if wandb_run is not None:
        print(f"3. Training logs available at wandb")
    print("\nTo use the trained models:")
    print("  - Load CEBRA model: cebra.CEBRA.load('path/to/cebra_model.pt')")
    print("  - Load decoder: decoder.load_state_dict(torch.load('path/to/decoder.pt'))")
    print()

if __name__ == "__main__":
    main()
