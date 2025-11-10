#!/usr/bin/env python3
"""
Main script to train CEBRA on Allen Brain Observatory data
and decode natural movie frames using HalfUNet

Usage:
    # Training mode
    python run_allen_cebra.py --config allen_config.yaml
    python run_allen_cebra.py --config allen_config.yaml --data_dir /path/to/data

    # Test only mode (load pretrained models and evaluate)
    python run_allen_cebra.py --config allen_config.yaml --test_only \
        --pretrained_cebra_dir results/cebra_models_20231201_120000 \
        --decoder_checkpoint checkpoints/best_model.pt
"""

import argparse
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Import CEBRA decoders
from cebra.integrations.decoders import SingleLayerDecoder, TwoLayersDecoder

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

def create_kirby_dataset(config, split='train', pretrain=False, finetune=False):
    """
    Create Kirby Dataset for Allen Brain Observatory

    Args:
        config: Configuration dictionary
        split: 'train', 'valid', or 'test'
        pretrain: Whether to use pretrain mode (combines all splits)
        finetune: Whether to use finetune mode (filters specific cell types)

    Returns:
        KirbyDataset instance
    """
    print(f"\n{'='*80}")
    print(f"Creating Kirby Dataset ({split} split)")
    if pretrain:
        print("  Mode: PRETRAIN (all splits combined)")
    elif finetune:
        print("  Mode: FINETUNE (filtered cell types)")
    else:
        print("  Mode: FROM SCRATCH")
    print(f"{'='*80}")

    # Prepare include configuration
    include = [{
        'selection': [{
            'dandiset': 'allen_brain_observatory_calcium'
        }]
    }]

    # Get task from config (default to movie_decoding_one for backward compatibility)
    task = config.get('task', 'movie_decoding_one')

    # Create dataset
    dataset = KirbyDataset(
        root=config['dataset']['data_dir'],
        split=split,
        include=include,
        transform=None,
        pretrain=pretrain,
        finetune=finetune,
        small_model=config.get('small_model', False),
        task=task,
        ssl_mode=config.get('ssl_mode', 'predictable'),
        model_dim=config['model']['output_dimension']
    )

    # Disable data leakage check for Allen dataset
    # Allen dataset uses time-based splits which don't align with mask-based checks
    dataset._check_for_data_leakage_flag = False

    print(f"✅ Dataset created with {len(dataset)} sessions")
    return dataset

def load_pretrained_cebra_models(pretrained_dir, split='train'):
    """
    Load pretrained CEBRA models from directory

    Args:
        pretrained_dir: Directory containing pretrained CEBRA models
        split: Which split to load ('train', 'valid', or 'test')

    Returns:
        Dictionary mapping session_id to loaded CEBRA model
    """
    if not CEBRA_AVAILABLE:
        raise ImportError("CEBRA is not available. Please install it.")

    pretrained_dir = Path(pretrained_dir)
    split_dir = pretrained_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Pretrained split directory not found: {split_dir}")

    print(f"\n{'='*80}")
    print(f"Loading Pretrained CEBRA Models ({split} split)")
    print(f"{'='*80}")
    print(f"Loading from: {split_dir}")

    cebra_models = {}
    model_files = list(split_dir.glob("cebra_*.pt"))

    if len(model_files) == 0:
        raise FileNotFoundError(f"No CEBRA model files found in {split_dir}")

    print(f"Found {len(model_files)} pretrained CEBRA models")

    for model_path in tqdm(model_files, desc=f"Loading CEBRA models ({split})"):
        # Extract session_id from filename (e.g., cebra_session123.pt -> session123)
        session_id = model_path.stem.replace("cebra_", "")
        # Restore original session_id format (e.g., allen_brain_observatory_calcium_657776356 -> allen_brain_observatory_calcium/657776356)
        session_id = session_id.replace("allen_brain_observatory_calcium_", "allen_brain_observatory_calcium/")

        try:
            # Load CEBRA model
            cebra_model = cebra.CEBRA.load(str(model_path))
            cebra_models[session_id] = cebra_model
        except Exception as e:
            print(f"  ⚠️  Error loading CEBRA model for session {session_id}: {e}")
            continue

    print(f"✅ Loaded {len(cebra_models)} pretrained CEBRA models")
    return cebra_models


def train_single_cebra(session_id, dataset, config, device_id):
    """
    Train CEBRA for a single session on a specific GPU

    Args:
        session_id: Session identifier
        dataset: Dataset object
        config: Configuration dictionary
        device_id: GPU device ID

    Returns:
        Tuple of (session_id, trained CEBRA model) or (session_id, None) on error
    """
    try:
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

        session_data = dataset.get_session_data(session_id)
        neural_data = extract_neural_data(session_data)

        # Create and train CEBRA model for this session
        cebra_model = cebra.CEBRA(
            model_architecture=config['model']['model_architecture'],
            batch_size=config['model']['batch_size'],
            learning_rate=config['model']['learning_rate'],
            max_iterations=config['model']['max_iterations'],
            output_dimension=config['model']['output_dimension'],
            device=device,
            verbose=False,
        )

        # Train on this session
        cebra_model.fit(neural_data)

        return (session_id, cebra_model)

    except Exception as e:
        print(f"  ⚠️  Error training CEBRA for session {session_id}: {e}")
        return (session_id, None)


def train_cebra_per_session(config, dataset, split='train', device='cuda', num_gpus=None, pretrained_models=None):
    """
    Train CEBRA model separately for each session (to handle different neuron counts)
    Supports multi-GPU parallel training

    Args:
        config: Configuration dictionary
        dataset: Dataset object
        split: 'train', 'valid', or 'test'
        device: Device to use ('cuda' or 'cpu')
        num_gpus: Number of GPUs to use (None = use all available, 1 = sequential)
        pretrained_models: Dictionary of pretrained CEBRA models (optional)

    Returns:
        Dictionary mapping session_id to trained CEBRA model
    """
    if not CEBRA_AVAILABLE:
        raise ImportError("CEBRA is not available. Please install it.")

    print(f"\n{'='*80}")
    print(f"Training CEBRA Models Per Session ({split} split)")
    print(f"{'='*80}")

    # Check if pretrained models are provided
    if pretrained_models is not None:
        print(f"Using {len(pretrained_models)} pretrained CEBRA models")
        print("Will only train models for sessions without pretrained weights")

    # Determine number of GPUs to use
    if device == 'cpu':
        num_gpus = 0
    elif num_gpus is None:
        num_gpus = torch.cuda.device_count()

    num_gpus = max(0, num_gpus)  # Ensure non-negative

    if num_gpus == 0:
        print("Training on CPU (sequential)")
    elif num_gpus == 1:
        print("Training on single GPU (sequential)")
    else:
        print(f"Training on {num_gpus} GPUs (parallel)")

    session_cebra_models = {}
    session_ids = dataset.session_ids

    # Separate sessions into pretrained and need-training
    sessions_to_train = []
    if pretrained_models is not None:
        for session_id in session_ids:
            if session_id in pretrained_models:
                # Use pretrained model
                session_cebra_models[session_id] = pretrained_models[session_id]
            else:
                # Need to train this session
                sessions_to_train.append(session_id)
        print(f"Using {len(session_cebra_models)} pretrained models")
        print(f"Will train {len(sessions_to_train)} new models")
    else:
        sessions_to_train = session_ids
        print(f"Training {len(sessions_to_train)} models from scratch")

    # Skip training if no sessions need training
    if len(sessions_to_train) == 0:
        print(f"✅ All sessions use pretrained models, no training needed")
        return session_cebra_models

    # Multi-GPU parallel training
    if num_gpus > 1:
        # Distribute sessions across GPUs
        session_gpu_map = [(sid, i % num_gpus) for i, sid in enumerate(sessions_to_train)]

        print(f"Distributing {len(sessions_to_train)} sessions across {num_gpus} GPUs...")

        # Process sessions in parallel
        with tqdm(total=len(sessions_to_train), desc=f"Training CEBRA ({split})") as pbar:
            # Use threading for I/O bound tasks (each GPU handles its own sessions)
            from concurrent.futures import ThreadPoolExecutor

            def train_on_gpu(session_id, gpu_id):
                result = train_single_cebra(session_id, dataset, config, gpu_id)
                pbar.update(1)
                return result

            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [
                    executor.submit(train_on_gpu, session_id, gpu_id)
                    for session_id, gpu_id in session_gpu_map
                ]

                for future in as_completed(futures):
                    session_id, cebra_model = future.result()
                    if cebra_model is not None:
                        session_cebra_models[session_id] = cebra_model

    # Single GPU or CPU (sequential)
    else:
        device_id = 0 if num_gpus == 1 else 'cpu'
        for session_id in tqdm(sessions_to_train, desc=f"Training CEBRA ({split})"):
            session_id_result, cebra_model = train_single_cebra(
                session_id, dataset, config, device_id
            )
            if cebra_model is not None:
                session_cebra_models[session_id] = cebra_model

    print(f"\n✅ Trained CEBRA for {len(session_cebra_models)}/{len(session_ids)} sessions")
    return session_cebra_models

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

    # Safety check for data validity
    if neural_data is None or len(neural_data) == 0:
        raise ValueError("Neural data is empty or None")

    # Check for NaN or Inf values
    if np.any(np.isnan(neural_data)) or np.any(np.isinf(neural_data)):
        print("Warning: NaN or Inf detected in neural data. Replacing with zeros.")
        neural_data = np.nan_to_num(neural_data, nan=0.0, posinf=0.0, neginf=0.0)

    return neural_data


def extract_movie_frames(session_data, dataset):
    """
    Extract movie frames from session_data and dataset.

    For Allen dataset, frame numbers are stored in natural_movie_one.frame_number,
    and actual frames are in dataset.movie_frames.

    Args:
        session_data: Data object from dataset
        dataset: The dataset object (has movie_frames attribute)

    Returns:
        torch.Tensor of movie frames, shape (T, H, W)
    """
    # Check if movie_frames directly exists (from __getitem__)
    if hasattr(session_data, 'movie_frames'):
        return session_data.movie_frames

    # For Allen dataset, extract from natural_movie_one
    if hasattr(session_data, 'natural_movie_one'):
        natural_movie = session_data.natural_movie_one

        # Get frame indices
        if hasattr(natural_movie, 'frame_number'):
            frame_indices_raw = natural_movie.frame_number

            # Convert to numpy if needed
            if hasattr(frame_indices_raw, 'cpu'):
                frame_indices = frame_indices_raw.cpu().numpy()
            elif hasattr(frame_indices_raw, 'numpy'):
                frame_indices = frame_indices_raw.numpy()
            else:
                frame_indices = frame_indices_raw

            # Convert to integers
            frame_indices = frame_indices.squeeze().astype(int)

            # Get actual movie frames from dataset
            if hasattr(dataset, 'movie_frames'):
                # CRITICAL: Validate frame indices to prevent out of bounds access
                max_frames = len(dataset.movie_frames)

                # Clamp indices to valid range [0, max_frames-1]
                frame_indices = np.clip(frame_indices, 0, max_frames - 1)

                # Additional safety check
                if np.any(frame_indices < 0) or np.any(frame_indices >= max_frames):
                    print(f"Warning: Frame indices out of bounds detected. Max frames: {max_frames}")
                    print(f"  Min index: {frame_indices.min()}, Max index: {frame_indices.max()}")
                    frame_indices = np.clip(frame_indices, 0, max_frames - 1)

                movie_frames = dataset.movie_frames[frame_indices, :, :]
                return torch.from_numpy(movie_frames).float()
            else:
                raise AttributeError(f"Dataset does not have movie_frames attribute")
        else:
            raise AttributeError(f"natural_movie_one does not have frame_number attribute")

    raise AttributeError(f"Cannot extract movie frames. Available attributes: {session_data.keys}")

def extract_drifting_gratings_labels(session_data):
    """
    Extract drifting gratings orientation labels from session_data.

    For Allen dataset, orientation is stored in drifting_gratings.orientation,
    with 8 possible orientations (0, 45, 90, 135, 180, 225, 270, 315 degrees).

    Args:
        session_data: Data object from dataset

    Returns:
        torch.Tensor of orientation labels, shape (T,) with values in [0-7]
    """
    # Check if drifting_gratings directly exists
    if hasattr(session_data, 'drifting_gratings'):
        drifting_gratings = session_data.drifting_gratings

        # Get orientation labels
        if hasattr(drifting_gratings, 'orientation'):
            orientation_raw = drifting_gratings.orientation

            # Convert to numpy if needed
            if hasattr(orientation_raw, 'cpu'):
                orientation = orientation_raw.cpu().numpy()
            elif hasattr(orientation_raw, 'numpy'):
                orientation = orientation_raw.numpy()
            else:
                orientation = orientation_raw

            # Convert to integers (should be 0-7)
            orientation = orientation.squeeze().astype(int)

            # Validate orientation values are in valid range [0, 7]
            if np.any(orientation < 0) or np.any(orientation > 7):
                print(f"Warning: Orientation values out of range. Min: {orientation.min()}, Max: {orientation.max()}")
                orientation = np.clip(orientation, 0, 7)

            return torch.from_numpy(orientation).long()
        else:
            raise AttributeError(f"drifting_gratings does not have orientation attribute")

    raise AttributeError(f"Cannot extract drifting gratings labels. Available attributes: {session_data.keys}")

def train_decoder_classification(config, train_cebra_models, valid_cebra_models, train_dataset, valid_dataset, device='cuda', wandb_run=None, start_epoch=0, checkpoint_path=None):
    """
    Train CEBRA decoder (SingleLayer or TwoLayers) for classification tasks

    This is for tasks like drifting gratings orientation classification.
    Uses standard CEBRA decoders instead of HalfUNet.

    Args:
        config: Configuration dictionary
        train_cebra_models: Dictionary of session_id -> trained CEBRA model (train)
        valid_cebra_models: Dictionary of session_id -> trained CEBRA model (valid)
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        device: Device to use for training
        wandb_run: Weights & Biases run object for logging
        start_epoch: Starting epoch (for resume training)
        checkpoint_path: Path to checkpoint directory for saving models

    Returns:
        Trained CEBRA decoder
    """
    print(f"\n{'='*80}")
    print("Stage 2: Training CEBRA Decoder (Classification)")
    print(f"{'='*80}")
    print("Using pre-trained CEBRA embeddings for decoder training")

    # Get decoder configuration
    decoder_config = config.get('decoder', {})
    decoder_type = decoder_config.get('type', 'TwoLayersDecoder')
    output_dim = decoder_config.get('output_dim', 8)  # 8 classes for drifting gratings

    # Initialize decoder
    input_dim = config['model']['output_dimension']
    if decoder_type == 'SingleLayerDecoder':
        decoder = SingleLayerDecoder(input_dim=input_dim, output_dim=output_dim)
    elif decoder_type == 'TwoLayersDecoder':
        decoder = TwoLayersDecoder(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    print(f"\nDecoder: {decoder_type}")
    print(f"  Input dim: {input_dim} (CEBRA embedding)")
    print(f"  Output dim: {output_dim} (number of classes)")

    # Move to device and enable DataParallel if multiple GPUs available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"\n🚀 Using DataParallel with {torch.cuda.device_count()} GPUs")
        decoder = nn.DataParallel(decoder)
        decoder = decoder.to(device)
    else:
        decoder = decoder.to(device)

    # Loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer for decoder only (CEBRA is frozen)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=decoder_config.get('learning_rate', 0.001)
    )

    print(f"\nDecoder optimizer created")
    print(f"  CEBRA models: FROZEN (pre-trained)")
    print(f"  Decoder: TRAINABLE")

    # Training parameters
    num_epochs = decoder_config.get('num_epochs', 50)
    batch_size = decoder_config.get('batch_size', 128)

    print(f"Training parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")

    # Setup checkpoint directory
    if checkpoint_path is None:
        checkpoint_path = Path(config['output']['save_dir']) / 'checkpoints'
    else:
        checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_path}")

    # Load checkpoint if resuming training
    best_valid_loss = float('inf')
    best_valid_accuracy = 0.0
    if start_epoch > 0:
        last_checkpoint = checkpoint_path / 'last_model.pt'
        if last_checkpoint.exists():
            print(f"\n🔄 Loading checkpoint to resume from epoch {start_epoch+1}/{num_epochs}")
            checkpoint = torch.load(last_checkpoint, weights_only=False)

            # Load model state
            if isinstance(decoder, nn.DataParallel):
                decoder.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                decoder.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load best metrics
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
            best_valid_accuracy = checkpoint.get('best_valid_accuracy', 0.0)

            print(f"  ✅ Checkpoint loaded successfully")
            print(f"  Best validation loss: {best_valid_loss:.4f}")
            print(f"  Best validation accuracy: {best_valid_accuracy:.4f}")
        else:
            print(f"\n⚠️  Warning: No checkpoint found at {last_checkpoint}")
            print("  Starting training from scratch")
            start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        decoder.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        # Training phase
        for session_id in tqdm(train_dataset.session_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                # Skip if no CEBRA model for this session
                if session_id not in train_cebra_models:
                    continue

                session_data = train_dataset.get_session_data(session_id)

                # Get neural data and labels
                neural_data = extract_neural_data(session_data)
                labels = extract_drifting_gratings_labels(session_data)  # Shape: (T,)

                # Validate data shapes
                if len(neural_data) != len(labels):
                    print(f"Warning: Length mismatch for session {session_id}. Neural: {len(neural_data)}, Labels: {len(labels)}. Taking minimum.")
                    min_len = min(len(neural_data), len(labels))
                    neural_data = neural_data[:min_len]
                    labels = labels[:min_len]

                # Get pre-trained CEBRA model for this session
                cebra_model = train_cebra_models[session_id]

                # Generate embeddings (no gradients - CEBRA is frozen)
                with torch.no_grad():
                    embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

                # Move labels to device
                targets = labels.to(device)

                # Mini-batch training
                num_samples = len(embeddings)

                if num_samples == 0:
                    print(f"Warning: No samples for session {session_id}. Skipping.")
                    continue

                indices = torch.randperm(num_samples, device=embeddings.device)

                for i in range(0, num_samples, batch_size):
                    try:
                        end_idx = min(i + batch_size, num_samples)
                        batch_indices = indices[i:end_idx]

                        if len(batch_indices) == 0:
                            continue

                        batch_embeddings = embeddings[batch_indices]
                        batch_targets = targets[batch_indices]

                        # Forward pass
                        predictions = decoder(batch_embeddings)

                        # Compute loss
                        loss = criterion(predictions, batch_targets)

                        # Check for NaN/Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN/Inf detected in loss for session {session_id}, batch {i}. Skipping batch.")
                            continue

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

                        optimizer.step()

                        # Track losses and accuracy
                        train_losses.append(loss.item())

                        # Calculate accuracy
                        _, predicted = torch.max(predictions.data, 1)
                        train_total += batch_targets.size(0)
                        train_correct += (predicted == batch_targets).sum().item()

                    except (IndexError, RuntimeError) as e:
                        print(f"Warning: Error in session {session_id}, batch {i}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        continue

            except Exception as e:
                print(f"Warning: Error processing session {session_id}: {e}")
                continue

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0

        # Validation phase
        decoder.eval()
        valid_losses = []
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for session_id in valid_dataset.session_ids:
                try:
                    # Skip if no CEBRA model for this session
                    if session_id not in valid_cebra_models:
                        continue

                    session_data = valid_dataset.get_session_data(session_id)

                    neural_data = extract_neural_data(session_data)
                    labels = extract_drifting_gratings_labels(session_data)

                    # Validate data shapes
                    if len(neural_data) != len(labels):
                        print(f"Warning: Length mismatch in validation for session {session_id}. Taking minimum.")
                        min_len = min(len(neural_data), len(labels))
                        neural_data = neural_data[:min_len]
                        labels = labels[:min_len]

                    if len(neural_data) == 0:
                        continue

                    # Get pre-trained CEBRA model for this session
                    cebra_model = valid_cebra_models[session_id]

                    # Generate embeddings
                    embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)
                    targets = labels.to(device)

                    # Forward pass
                    predictions = decoder(embeddings)

                    # Compute loss
                    loss = criterion(predictions, targets)
                    valid_losses.append(loss.item())

                    # Calculate accuracy
                    _, predicted = torch.max(predictions.data, 1)
                    valid_total += targets.size(0)
                    valid_correct += (predicted == targets).sum().item()

                except Exception as e:
                    print(f"Warning: Error in validation session {session_id}: {e}")
                    continue

        avg_valid_loss = np.mean(valid_losses) if valid_losses else float('inf')
        valid_accuracy = 100 * valid_correct / valid_total if valid_total > 0 else 0

        # Print summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.2f}% | Valid Accuracy: {valid_accuracy:.2f}%")

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'train/accuracy': train_accuracy,
                'valid/loss': avg_valid_loss,
                'valid/accuracy': valid_accuracy,
                'average_val_metric': valid_accuracy,
            })

        # Prepare checkpoint dictionary
        model_state = decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict()
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': avg_valid_loss,
            'valid_accuracy': valid_accuracy,
            'best_valid_loss': best_valid_loss,
            'best_valid_accuracy': best_valid_accuracy,
            'config': config
        }

        # Save last model (every epoch)
        last_checkpoint_path = checkpoint_path / 'last_model.pt'
        torch.save(checkpoint, last_checkpoint_path)

        # Save best model (based on accuracy)
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_valid_loss = avg_valid_loss
            best_checkpoint_path = checkpoint_path / 'best_model.pt'
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  ✅ New best model saved! (Accuracy: {best_valid_accuracy:.2f}%)")

    print("✅ Decoder training complete!")
    print(f"Best validation accuracy: {best_valid_accuracy:.2f}%")
    return decoder

def train_decoder_only(config, train_cebra_models, valid_cebra_models, train_dataset, valid_dataset, device='cuda', wandb_run=None, start_epoch=0, checkpoint_path=None):
    """
    Train HalfUNet decoder using pre-trained CEBRA embeddings

    Two-stage approach:
    Stage 1: Train CEBRA per session (handles different neuron counts)
    Stage 2: Train HalfUNet decoder on embeddings (all same dimension)

    Args:
        config: Configuration dictionary
        train_cebra_models: Dictionary of session_id -> trained CEBRA model (train)
        valid_cebra_models: Dictionary of session_id -> trained CEBRA model (valid)
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        device: Device to use for training
        wandb_run: Weights & Biases run object for logging
        start_epoch: Starting epoch (for resume training)
        checkpoint_path: Path to checkpoint directory for saving models

    Returns:
        Trained HalfUNet decoder
    """
    print(f"\n{'='*80}")
    print("Stage 2: Training HalfUNet Decoder")
    print(f"{'='*80}")
    print("Using pre-trained CEBRA embeddings for decoder training")

    # Initialize HalfUNet decoder
    decoder = HalfUNet(
        in_channels=1,
        out_channels=1,  # Grayscale movie frames
        latent_dim=config['model']['output_dimension']  # Same as CEBRA output dim
    )

    # Move to device and enable DataParallel if multiple GPUs available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"\n🚀 Using DataParallel with {torch.cuda.device_count()} GPUs for HalfUNet")
        decoder = nn.DataParallel(decoder)
        decoder = decoder.to(device)
    else:
        decoder = decoder.to(device)

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

    # Optimizer for decoder only (CEBRA is frozen)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=config.get('decoder', {}).get('learning_rate', 0.001)
    )

    print(f"\nDecoder optimizer created")
    print(f"  CEBRA models: FROZEN (pre-trained)")
    print(f"  HalfUNet: TRAINABLE")

    # Training parameters
    num_epochs = config.get('decoder', {}).get('num_epochs', 50)
    batch_size = config.get('decoder', {}).get('batch_size', 32)

    print(f"Training parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")

    # Setup checkpoint directory
    if checkpoint_path is None:
        checkpoint_path = Path(config['output']['save_dir']) / 'checkpoints'
    else:
        checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_path}")

    # Load checkpoint if resuming training
    best_valid_loss = float('inf')
    best_valid_ssim = 0.0
    if start_epoch > 0:
        last_checkpoint = checkpoint_path / 'last_model.pt'
        if last_checkpoint.exists():
            print(f"\n🔄 Loading checkpoint to resume from epoch {start_epoch+1}/{num_epochs}")
            checkpoint = torch.load(last_checkpoint, weights_only=False)

            # Load model state
            if isinstance(decoder, nn.DataParallel):
                decoder.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                decoder.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load best metrics
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
            best_valid_ssim = checkpoint.get('best_valid_ssim', 0.0)

            print(f"  ✅ Checkpoint loaded successfully")
            print(f"  Best validation loss: {best_valid_loss:.4f}")
            print(f"  Best validation SSIM: {best_valid_ssim:.4f}")
        else:
            print(f"\n⚠️  Warning: No checkpoint found at {last_checkpoint}")
            print("  Starting training from scratch")
            start_epoch = 0

    # Training loop

    for epoch in range(start_epoch, num_epochs):
        decoder.train()
        train_losses = []
        train_loss_details = {
            'l1': [], 'ssim': [], 'perceptual': [],
            'gradient': [], 'focal': [], 'fft': []
        }

        # Training phase
        for session_id in tqdm(train_dataset.session_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                # Skip if no CEBRA model for this session
                if session_id not in train_cebra_models:
                    continue

                session_data = train_dataset.get_session_data(session_id)

                # Get neural data and movie frames
                neural_data = extract_neural_data(session_data)
                movie_frames = extract_movie_frames(session_data, train_dataset)  # Shape: (T, H, W)

                # Validate data shapes
                if len(neural_data) != len(movie_frames):
                    print(f"Warning: Length mismatch for session {session_id}. Neural: {len(neural_data)}, Frames: {len(movie_frames)}. Taking minimum.")
                    min_len = min(len(neural_data), len(movie_frames))
                    neural_data = neural_data[:min_len]
                    movie_frames = movie_frames[:min_len]

                # Get pre-trained CEBRA model for this session
                cebra_model = train_cebra_models[session_id]

                # Generate embeddings (no gradients - CEBRA is frozen)
                with torch.no_grad():
                    embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

                # Prepare movie frames with normalization
                targets = movie_frames.unsqueeze(1).float().to(device)  # Add channel dim

                # Normalize to [0, 1] range if needed
                if targets.max() > 1.0:
                    targets = targets / 255.0

                # Clip to valid range
                targets = torch.clamp(targets, 0.0, 1.0)

                # Mini-batch training
                num_samples = len(embeddings)

                # Safety check: ensure we have valid samples
                if num_samples == 0:
                    print(f"Warning: No samples for session {session_id}. Skipping.")
                    continue

                # Ensure embeddings and targets have matching sizes
                if len(embeddings) != len(targets):
                    print(f"Warning: Size mismatch for session {session_id}. Embeddings: {len(embeddings)}, Targets: {len(targets)}. Skipping.")
                    continue

                indices = torch.randperm(num_samples, device=embeddings.device)

                for i in range(0, num_samples, batch_size):
                    try:
                        # Ensure batch_indices doesn't exceed bounds
                        end_idx = min(i + batch_size, num_samples)
                        batch_indices = indices[i:end_idx]

                        # Skip if batch is too small (optional, but helps with stability)
                        if len(batch_indices) == 0:
                            continue

                        # Verify indices are within bounds
                        if torch.max(batch_indices) >= num_samples or torch.min(batch_indices) < 0:
                            print(f"Warning: Invalid batch indices detected in session {session_id}, batch {i}. Max: {torch.max(batch_indices)}, Min: {torch.min(batch_indices)}, Num samples: {num_samples}")
                            continue

                        # Safe indexing with bounds check
                        batch_embeddings = embeddings[batch_indices]
                        batch_targets = targets[batch_indices]

                        # Forward pass
                        predictions = decoder(batch_embeddings)

                        # Ensure predictions have channel dimension
                        if predictions.dim() == 3:
                            predictions = predictions.unsqueeze(1)

                        # Clamp predictions to valid range
                        predictions = torch.clamp(predictions, 0.0, 1.0)

                    except (IndexError, RuntimeError) as e:
                        print(f"Warning: Indexing/forward error in session {session_id}, batch {i}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        continue

                    # Compute all 6 losses with error handling
                    try:
                        loss_l1_val = l1_loss(predictions, batch_targets)
                        loss_ssim_val = ssim_loss(predictions, batch_targets)
                        loss_perceptual_val = perceptual_loss(predictions, batch_targets)
                        loss_gradient_val = gradient_loss(predictions, batch_targets)
                        loss_focal_val = focal_loss(predictions, batch_targets)
                        loss_fft_val = fft_loss(predictions, batch_targets)

                        # Check each loss for NaN/Inf
                        losses_dict = {
                            'l1': loss_l1_val, 'ssim': loss_ssim_val,
                            'perceptual': loss_perceptual_val, 'gradient': loss_gradient_val,
                            'focal': loss_focal_val, 'fft': loss_fft_val
                        }

                        has_invalid = False
                        for loss_name, loss_val in losses_dict.items():
                            if torch.isnan(loss_val) or torch.isinf(loss_val):
                                print(f"Warning: NaN/Inf detected in {loss_name} loss for session {session_id}, batch {i}")
                                has_invalid = True

                        if has_invalid:
                            print(f"Skipping batch due to invalid loss values")
                            continue

                        # Weighted combination of losses
                        total_loss = (
                            loss_weights['l1'] * loss_l1_val +
                            loss_weights['ssim'] * loss_ssim_val +
                            loss_weights['perceptual'] * loss_perceptual_val +
                            loss_weights['gradient'] * loss_gradient_val +
                            loss_weights['focal'] * loss_focal_val +
                            loss_weights['fft'] * loss_fft_val
                        )

                        # Final check for NaN/Inf in total loss
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            print(f"Warning: NaN/Inf detected in total loss for session {session_id}, batch {i}. Skipping batch.")
                            continue

                    except Exception as e:
                        print(f"Warning: Error computing loss for session {session_id}, batch {i}: {e}")
                        continue

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()

                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

                    optimizer.step()

                    # Track losses
                    train_losses.append(total_loss.item())
                    train_loss_details['l1'].append(loss_l1_val.item())
                    train_loss_details['ssim'].append(loss_ssim_val.item())
                    train_loss_details['perceptual'].append(loss_perceptual_val.item())
                    train_loss_details['gradient'].append(loss_gradient_val.item())
                    train_loss_details['focal'].append(loss_focal_val.item())
                    train_loss_details['fft'].append(loss_fft_val.item())

            except RuntimeError as e:
                if 'CUDA' in str(e) or 'out of bounds' in str(e):
                    print(f"⚠️  CUDA error in session {session_id}: {e}")
                    print(f"   Synchronizing CUDA and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                continue
            except Exception as e:
                print(f"Warning: Error processing session {session_id}: {e}")
                continue

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')

        # Validation phase
        decoder.eval()
        valid_losses = []
        valid_loss_details = {
            'l1': [], 'ssim': [], 'perceptual': [],
            'gradient': [], 'focal': [], 'fft': []
        }
        valid_ssim_scores = []  # For tracking SSIM metric (not loss)

        # For wandb image logging (save first batch of predictions)
        first_predictions = None
        first_targets = None

        with torch.no_grad():
            for idx, session_id in enumerate(valid_dataset.session_ids):
                try:
                    # Skip if no CEBRA model for this session
                    if session_id not in valid_cebra_models:
                        continue

                    session_data = valid_dataset.get_session_data(session_id)

                    neural_data = extract_neural_data(session_data)
                    movie_frames = extract_movie_frames(session_data, valid_dataset)

                    # Validate data shapes
                    if len(neural_data) != len(movie_frames):
                        print(f"Warning: Length mismatch in validation for session {session_id}. Neural: {len(neural_data)}, Frames: {len(movie_frames)}. Taking minimum.")
                        min_len = min(len(neural_data), len(movie_frames))
                        neural_data = neural_data[:min_len]
                        movie_frames = movie_frames[:min_len]

                    # Get pre-trained CEBRA model for this session
                    cebra_model = valid_cebra_models[session_id]

                    # Generate embeddings
                    embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

                    # Prepare targets with normalization
                    targets = movie_frames.unsqueeze(1).float().to(device)

                    # Normalize to [0, 1] range if needed
                    if targets.max() > 1.0:
                        targets = targets / 255.0

                    # Clip to valid range
                    targets = torch.clamp(targets, 0.0, 1.0)

                    # Safety check: ensure embeddings and targets match
                    if len(embeddings) != len(targets):
                        print(f"Warning: Size mismatch in validation for session {session_id}. Embeddings: {len(embeddings)}, Targets: {len(targets)}. Skipping.")
                        continue

                    if len(embeddings) == 0:
                        print(f"Warning: No samples in validation for session {session_id}. Skipping.")
                        continue

                    predictions = decoder(embeddings)

                    # Ensure predictions have channel dimension
                    if predictions.dim() == 3:
                        predictions = predictions.unsqueeze(1)

                    # Clamp predictions to valid range
                    predictions = torch.clamp(predictions, 0.0, 1.0)

                    # Compute all 6 losses
                    loss_l1_val = l1_loss(predictions, targets)
                    loss_ssim_val = ssim_loss(predictions, targets)
                    loss_perceptual_val = perceptual_loss(predictions, targets)
                    loss_gradient_val = gradient_loss(predictions, targets)
                    loss_focal_val = focal_loss(predictions, targets)
                    loss_fft_val = fft_loss(predictions, targets)

                    # Calculate SSIM score (metric, not loss)
                    # SSIM score ranges from -1 to 1, where 1 means perfect similarity
                    # SSIMLoss typically returns (1 - SSIM), so we convert it back
                    ssim_score = 1.0 - loss_ssim_val.item()
                    valid_ssim_scores.append(ssim_score)

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

                except RuntimeError as e:
                    if 'CUDA' in str(e) or 'out of bounds' in str(e):
                        print(f"⚠️  CUDA error in validation session {session_id}: {e}")
                        print(f"   Synchronizing CUDA and continuing...")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    continue
                except Exception as e:
                    continue

        avg_valid_loss = np.mean(valid_losses) if valid_losses else float('inf')
        avg_valid_ssim = np.mean(valid_ssim_scores) if valid_ssim_scores else 0.0

        # Calculate average for each loss component
        avg_train_loss_details = {k: np.mean(v) if v else 0.0 for k, v in train_loss_details.items()}
        avg_valid_loss_details = {k: np.mean(v) if v else 0.0 for k, v in valid_loss_details.items()}

        # Print summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        print(f"  Validation SSIM Score: {avg_valid_ssim:.4f}")
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
                'average_val_metric': avg_valid_ssim,  # SSIM score for validation

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

        # Prepare checkpoint dictionary
        model_state = decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict()
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': avg_valid_loss,
            'valid_ssim': avg_valid_ssim,
            'best_valid_loss': best_valid_loss,
            'best_valid_ssim': best_valid_ssim,
            'config': config
        }

        # Save last model (every epoch)
        last_checkpoint_path = checkpoint_path / 'last_model.pt'
        torch.save(checkpoint, last_checkpoint_path)

        # Save best model (based on SSIM score)
        if avg_valid_ssim > best_valid_ssim:
            best_valid_ssim = avg_valid_ssim
            best_valid_loss = avg_valid_loss
            best_checkpoint_path = checkpoint_path / 'best_model.pt'
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  ✅ New best model saved! (SSIM: {best_valid_ssim:.4f})")

    print("✅ Decoder training complete!")
    print("HalfUNet decoder has been trained on frozen CEBRA embeddings")
    return decoder

def visualize_results(decoder, test_cebra_models, test_dataset, output_dir, device='cuda', num_samples=5):
    """
    Visualize reconstruction results

    Args:
        decoder: Trained HalfUNet decoder
        test_cebra_models: Dictionary of session_id -> trained CEBRA model (test)
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
        # Get first test session with CEBRA model
        session_id = None
        for sid in test_dataset.session_ids:
            if sid in test_cebra_models:
                session_id = sid
                break

        if session_id is None:
            print("⚠️  No test sessions with CEBRA models available")
            return None

        session_data = test_dataset.get_session_data(session_id)

        neural_data = extract_neural_data(session_data)
        movie_frames = extract_movie_frames(session_data, test_dataset)

        # Generate embeddings and predictions
        cebra_model = test_cebra_models[session_id]
        embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

        # Safety check
        if len(embeddings) == 0:
            print("⚠️  No embeddings available for visualization")
            return None

        predictions = decoder(embeddings).cpu()

        # Safety check for predictions
        if len(predictions) == 0:
            print("⚠️  No predictions available for visualization")
            return None

        # Select random samples (with bounds checking)
        num_available = len(predictions)
        num_to_sample = min(num_samples, num_available)

        if num_to_sample == 0:
            print("⚠️  No samples available for visualization")
            return None

        indices = np.random.choice(num_available, size=num_to_sample, replace=False)

        # Create visualization
        fig, axes = plt.subplots(num_to_sample, 2, figsize=(10, num_to_sample * 3))

        for i, idx in enumerate(indices):
            # Original frame
            if isinstance(movie_frames, torch.Tensor):
                frame_img = movie_frames[idx].cpu().numpy()
            else:
                frame_img = movie_frames[idx]
            axes[i, 0].imshow(frame_img, cmap='gray')
            axes[i, 0].set_title(f'Original Frame {idx}')
            axes[i, 0].axis('off')

            # Reconstructed frame
            pred_frame = predictions[idx].numpy()
            # If prediction is 1D (e.g., 256), reshape to 2D (e.g., 16x16)
            if pred_frame.ndim == 1:
                frame_size = int(np.sqrt(len(pred_frame)))
                pred_frame = pred_frame.reshape(frame_size, frame_size)
            axes[i, 1].imshow(pred_frame, cmap='gray')
            axes[i, 1].set_title(f'Reconstructed Frame {idx}')
            axes[i, 1].axis('off')

        plt.tight_layout()
        save_path = output_dir / 'reconstruction_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualization saved to {save_path}")

    return save_path

def save_test_results(decoder, test_cebra_models, test_dataset, output_dir, device='cuda'):
    """
    Save test ground truth and predictions to true.pt and pred.pt

    Args:
        decoder: Trained HalfUNet decoder
        test_cebra_models: Dictionary of session_id -> trained CEBRA model (test)
        test_dataset: Test dataset
        output_dir: Output directory for saving results
        device: Device to use
    """
    print(f"\n{'='*80}")
    print("Saving Test Results (Ground Truth and Predictions)")
    print(f"{'='*80}")

    decoder.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for session_id in tqdm(test_dataset.session_ids, desc="Generating test predictions"):
            try:
                # Skip if no CEBRA model for this session
                if session_id not in test_cebra_models:
                    print(f"  ⚠️  No CEBRA model for session {session_id}, skipping")
                    continue

                session_data = test_dataset.get_session_data(session_id)

                # Get neural data and movie frames
                neural_data = extract_neural_data(session_data)
                movie_frames = extract_movie_frames(session_data, test_dataset)

                # Validate data shapes
                if len(neural_data) != len(movie_frames):
                    print(f"  ⚠️  Length mismatch for session {session_id}. Neural: {len(neural_data)}, Frames: {len(movie_frames)}. Taking minimum.")
                    min_len = min(len(neural_data), len(movie_frames))
                    neural_data = neural_data[:min_len]
                    movie_frames = movie_frames[:min_len]

                if len(neural_data) == 0:
                    print(f"  ⚠️  No samples for session {session_id}, skipping")
                    continue

                # Get CEBRA model for this session
                cebra_model = test_cebra_models[session_id]

                # Generate embeddings
                embeddings = torch.from_numpy(cebra_model.transform(neural_data)).float().to(device)

                # Generate predictions
                predictions = decoder(embeddings)

                # Ensure predictions have channel dimension
                if predictions.dim() == 3:
                    predictions = predictions.unsqueeze(1)

                # Clamp predictions to valid range
                predictions = torch.clamp(predictions, 0.0, 1.0)

                # Prepare ground truth with same preprocessing as training
                ground_truth = movie_frames.unsqueeze(1).float()  # Add channel dim

                # Normalize to [0, 1] range if needed
                if ground_truth.max() > 1.0:
                    ground_truth = ground_truth / 255.0

                # Clip to valid range
                ground_truth = torch.clamp(ground_truth, 0.0, 1.0)

                # Move to CPU and collect
                all_predictions.append(predictions.cpu())
                all_ground_truths.append(ground_truth.cpu())

            except Exception as e:
                print(f"  ⚠️  Error processing session {session_id}: {e}")
                continue

    # Concatenate all results
    if len(all_predictions) == 0 or len(all_ground_truths) == 0:
        print("⚠️  No test results to save")
        return None, None

    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truths = torch.cat(all_ground_truths, dim=0)

    # Verify shapes match
    print(f"\n📊 Test Results Shape:")
    print(f"  Ground Truth: {all_ground_truths.shape}")
    print(f"  Predictions:  {all_predictions.shape}")

    if all_ground_truths.shape != all_predictions.shape:
        print(f"⚠️  Warning: Shape mismatch detected!")
        print(f"  Ground Truth: {all_ground_truths.shape}")
        print(f"  Predictions:  {all_predictions.shape}")
        # Try to fix by matching the minimum size
        min_size = min(len(all_ground_truths), len(all_predictions))
        all_ground_truths = all_ground_truths[:min_size]
        all_predictions = all_predictions[:min_size]
        print(f"  Adjusted to: {all_ground_truths.shape}")

    # Save to files
    true_path = output_dir / 'true.pt'
    pred_path = output_dir / 'pred.pt'

    torch.save(all_ground_truths, true_path)
    torch.save(all_predictions, pred_path)

    print(f"\n✅ Test results saved:")
    print(f"  Ground Truth: {true_path} (shape: {all_ground_truths.shape})")
    print(f"  Predictions:  {pred_path} (shape: {all_predictions.shape})")
    print(f"  Total samples: {len(all_ground_truths)}")

    return true_path, pred_path

def save_results(train_cebra_models, valid_cebra_models, test_cebra_models, decoder, config):
    """
    Save trained models and results

    Args:
        train_cebra_models: Dictionary of session_id -> trained CEBRA model (train)
        valid_cebra_models: Dictionary of session_id -> trained CEBRA model (valid)
        test_cebra_models: Dictionary of session_id -> trained CEBRA model (test)
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

    # Save CEBRA models per session
    cebra_dir = output_dir / f"cebra_models_{timestamp}"
    cebra_dir.mkdir(exist_ok=True)

    for split_name, models_dict in [('train', train_cebra_models), ('valid', valid_cebra_models), ('test', test_cebra_models)]:
        split_dir = cebra_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for session_id, cebra_model in models_dict.items():
            # Replace '/' with '_' in session_id to avoid directory creation issues
            safe_session_id = session_id.replace('/', '_')
            cebra_path = split_dir / f"cebra_{safe_session_id}.pt"
            cebra_model.save(str(cebra_path))
        print(f"   ✅ Saved {len(models_dict)} CEBRA models ({split_name}) to {split_dir}")

    # Save HalfUNet decoder
    decoder_path = output_dir / f"halfunet_decoder_{timestamp}.pt"
    torch.save(decoder.state_dict(), decoder_path)
    print(f"   ✅ Saved HalfUNet decoder to {decoder_path}")

    # Save metadata
    info = {
        'timestamp': timestamp,
        'cebra_output_dim': config['model']['output_dimension'],
        'task': 'movie_decoding_one',
        'num_train_sessions': len(train_cebra_models),
        'num_valid_sessions': len(valid_cebra_models),
        'num_test_sessions': len(test_cebra_models),
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
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for parallel CEBRA training (None = use all available, 1 = sequential)'
    )
    parser.add_argument(
        '--pretrain',
        action='store_true',
        help='Pretrain mode: use all splits combined (for stable/predictable cell types)'
    )
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Finetune mode: filter specific cell types based on ssl_mode'
    )
    parser.add_argument(
        '--pretrained_cebra_dir',
        type=str,
        default=None,
        help='Directory containing pretrained CEBRA models (e.g., results/cebra_models_20231201_120000)'
    )
    parser.add_argument(
        '--freeze_cebra',
        action='store_true',
        help='Freeze CEBRA models during finetune (only train decoder)'
    )
    parser.add_argument(
        '--continue_learning',
        action='store_true',
        help='Continue training from last checkpoint'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Directory containing checkpoints (default: output_dir/checkpoints)'
    )
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Test only mode: load pretrained models and evaluate on test set (requires --pretrained_cebra_dir and --decoder_checkpoint)'
    )
    parser.add_argument(
        '--decoder_checkpoint',
        type=str,
        default=None,
        help='Path to decoder checkpoint file (e.g., checkpoints/best_model.pt) for test_only mode'
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

    # Get pretrain/finetune settings from args or config
    pretrain = args.pretrain or config.get('pretrain', False)
    finetune = args.finetune or config.get('finetune', False)

    # Validate pretrain/finetune combination
    if pretrain and finetune:
        raise ValueError("Cannot use both --pretrain and --finetune at the same time")

    # Validate test_only mode requirements
    if args.test_only:
        if args.pretrained_cebra_dir is None:
            raise ValueError("--test_only requires --pretrained_cebra_dir to load pretrained CEBRA models")
        if args.decoder_checkpoint is None:
            raise ValueError("--test_only requires --decoder_checkpoint to load trained decoder")
        print("\n🧪 Mode: TEST ONLY")
        print("   Will load pretrained models and evaluate on test set only")

    # Print training mode
    if pretrain:
        print("\n🎯 Training Mode: PRETRAIN")
        print("   Using all splits combined with stable/predictable cell types")
    elif finetune:
        print("\n🎯 Training Mode: FINETUNE")
        print("   Using filtered cell types based on ssl_mode")
    else:
        print("\n🎯 Training Mode: FROM SCRATCH")
        print("   Using standard train/valid/test splits with all cell types")

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
    # In test_only mode, only create test dataset
    if args.test_only:
        train_dataset = None
        valid_dataset = None
        test_dataset = create_kirby_dataset(config, split='test', pretrain=pretrain, finetune=finetune)
    else:
        train_dataset = create_kirby_dataset(config, split='train', pretrain=pretrain, finetune=finetune)
        valid_dataset = create_kirby_dataset(config, split='valid', pretrain=pretrain, finetune=finetune)
        test_dataset = create_kirby_dataset(config, split='test', pretrain=pretrain, finetune=finetune)

    # Stage 1: Train or Load CEBRA per session
    if args.test_only:
        print("\n" + "="*80)
        print("Stage 1: Loading Pretrained CEBRA Models (Test Only)")
        print("="*80)

        # Load only test CEBRA models in test_only mode
        test_cebra_models = load_pretrained_cebra_models(args.pretrained_cebra_dir, split='test')
        train_cebra_models = None
        valid_cebra_models = None
    else:
        print("\n" + "="*80)
        print("Stage 1: Training/Loading CEBRA Models Per Session")
        print("="*80)
        print("Each session has different neuron counts - training separate CEBRA models")

        # Load pretrained CEBRA models if provided
        pretrained_train_models = None
        pretrained_valid_models = None
        pretrained_test_models = None

        if args.pretrained_cebra_dir is not None:
            print(f"\n🔄 Loading pretrained CEBRA models from: {args.pretrained_cebra_dir}")
            try:
                pretrained_train_models = load_pretrained_cebra_models(args.pretrained_cebra_dir, split='train')
                pretrained_valid_models = load_pretrained_cebra_models(args.pretrained_cebra_dir, split='valid')
                pretrained_test_models = load_pretrained_cebra_models(args.pretrained_cebra_dir, split='test')

                if args.freeze_cebra:
                    print("\n❄️  CEBRA models will be FROZEN (only decoder will be trained)")
                else:
                    print("\n🔥 CEBRA models will continue training (fine-tuning)")
            except Exception as e:
                print(f"\n⚠️  Error loading pretrained models: {e}")
                print("   Will train from scratch instead")
                pretrained_train_models = None
                pretrained_valid_models = None
                pretrained_test_models = None

        train_cebra_models = train_cebra_per_session(config, train_dataset, split='train', device=device, num_gpus=args.num_gpus, pretrained_models=pretrained_train_models)
        valid_cebra_models = train_cebra_per_session(config, valid_dataset, split='valid', device=device, num_gpus=args.num_gpus, pretrained_models=pretrained_valid_models)
        test_cebra_models = train_cebra_per_session(config, test_dataset, split='test', device=device, num_gpus=args.num_gpus, pretrained_models=pretrained_test_models)

    # Stage 2: Train or Load decoder
    # Determine which decoder to use based on task
    task = config.get('task', 'movie_decoding_one')

    if args.test_only:
        # Test only mode: load pretrained decoder
        print("\n" + "="*80)
        if task == 'drifting_gratings':
            print("Stage 2: Loading Pretrained CEBRA Decoder (Classification)")
        else:
            print("Stage 2: Loading Pretrained HalfUNet Decoder (Reconstruction)")
        print("="*80)

        # Load checkpoint
        checkpoint_path = Path(args.decoder_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")

        print(f"Loading decoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

        # Initialize decoder architecture based on task
        if task == 'drifting_gratings':
            decoder_config = config.get('decoder', {})
            decoder_type = decoder_config.get('type', 'TwoLayersDecoder')
            output_dim = decoder_config.get('output_dim', 8)
            input_dim = config['model']['output_dimension']

            if decoder_type == 'SingleLayerDecoder':
                decoder = SingleLayerDecoder(input_dim=input_dim, output_dim=output_dim)
            elif decoder_type == 'TwoLayersDecoder':
                decoder = TwoLayersDecoder(input_dim=input_dim, output_dim=output_dim)
            else:
                raise ValueError(f"Unknown decoder type: {decoder_type}")
        else:
            # HalfUNet decoder
            decoder = HalfUNet(
                in_channels=1,
                out_channels=1,
                latent_dim=config['model']['output_dimension']
            )

        # Load state dict
        decoder.load_state_dict(checkpoint['model_state_dict'])
        decoder = decoder.to(device)
        decoder.eval()

        print(f"✅ Loaded decoder from checkpoint")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Validation loss: {checkpoint.get('valid_loss', 'N/A')}")
        if task == 'drifting_gratings':
            print(f"   Validation accuracy: {checkpoint.get('valid_accuracy', 'N/A')}")
        else:
            print(f"   Validation SSIM: {checkpoint.get('valid_ssim', 'N/A')}")

    else:
        # Training mode
        print("\n" + "="*80)
        if task == 'drifting_gratings':
            print("Stage 2: Training CEBRA Decoder (Classification)")
        else:
            print("Stage 2: Training HalfUNet Decoder (Reconstruction)")
        print("="*80)
        print("All CEBRA embeddings have same dimension - can train single decoder")

        # Handle checkpoint resumption
        start_epoch = 0
        checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else Path(config['output']['save_dir']) / 'checkpoints'
        checkpoint_dir = Path(checkpoint_dir)

        if args.continue_learning:
            print("\n🔄 Attempting to resume from checkpoint...")
            last_checkpoint_path = checkpoint_dir / 'last_model.pt'
            if last_checkpoint_path.exists():
                checkpoint = torch.load(last_checkpoint_path, weights_only=False)
                completed_epoch = checkpoint['epoch']
                start_epoch = completed_epoch + 1  # Start from next epoch
                print(f"  ✅ Found checkpoint - last completed epoch: {completed_epoch+1}/{config['training']['max_epochs']}")
                print(f"  Will resume training from epoch {start_epoch+1}/{config['training']['max_epochs']}")
            else:
                print(f"  ⚠️  No checkpoint found at {last_checkpoint_path}")
                print("  Starting training from scratch")

        # Choose appropriate decoder based on task
        if task == 'drifting_gratings':
            decoder = train_decoder_classification(
                config,
                train_cebra_models,
                valid_cebra_models,
                train_dataset,
                valid_dataset,
                device=device,
                wandb_run=wandb_run,
                start_epoch=start_epoch,
                checkpoint_path=checkpoint_dir
            )
        else:
            decoder = train_decoder_only(
                config,
                train_cebra_models,
                valid_cebra_models,
                train_dataset,
                valid_dataset,
                device=device,
                wandb_run=wandb_run,
                start_epoch=start_epoch,
                checkpoint_path=checkpoint_dir
            )

    # Visualize results (only for reconstruction tasks)
    if task != 'drifting_gratings':
        visualize_results(
            decoder,
            test_cebra_models,
            test_dataset,
            config['output']['save_dir'],
            device=device
        )

        # Save test predictions and ground truth
        save_test_results(
            decoder,
            test_cebra_models,
            test_dataset,
            config['output']['save_dir'],
            device=device
        )
    else:
        print("\n" + "="*80)
        print("Skipping visualization (classification task)")
        print("="*80)

    # Save results (skip in test_only mode)
    if not args.test_only:
        save_results(train_cebra_models, valid_cebra_models, test_cebra_models, decoder, config)
    else:
        print("\n" + "="*80)
        print("Skipping model saving (test_only mode)")
        print("="*80)

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()
        print("✅ wandb run finished")

    print("\n" + "="*80)
    print("✅ All Done!")
    print("="*80)
    print("\nResults:")
    print(f"1. CEBRA models (per session) saved to {config['output']['save_dir']}/cebra_models_*/")
    print(f"2. HalfUNet decoder saved to {config['output']['save_dir']}/")
    print(f"3. Checkpoints saved to {config['output']['save_dir']}/checkpoints/ (best_model.pt, last_model.pt)")
    print(f"4. Test results saved to {config['output']['save_dir']}/ (true.pt, pred.pt)")
    print(f"5. Visualizations saved to {config['output']['save_dir']}/")
    if wandb_run is not None:
        print(f"6. Training logs available at wandb")
    print("\nTwo-Stage Training Summary:")
    print(f"  - Stage 1: Trained {len(train_cebra_models)} CEBRA models (one per session)")
    print(f"  - Stage 2: Trained single HalfUNet decoder on all embeddings")
    print("\nTo use the trained models:")
    print("  - Load CEBRA model: cebra.CEBRA.load('path/to/cebra_models_*/train/cebra_SESSION_ID.pt')")
    print("  - Load decoder: decoder.load_state_dict(torch.load('path/to/halfunet_decoder.pt'))")
    print("  - Load test results: torch.load('path/to/true.pt'), torch.load('path/to/pred.pt')")
    print("\nTo resume training:")
    print("  - python run_allen_cebra.py --config allen_config.yaml --continue_learning")
    print("\nTo test only (load pretrained models and evaluate):")
    print("  - python run_allen_cebra.py --config allen_config.yaml --test_only \\")
    print("      --pretrained_cebra_dir results/cebra_models_YYYYMMDD_HHMMSS \\")
    print("      --decoder_checkpoint checkpoints/best_model.pt")
    print()

if __name__ == "__main__":
    main()
