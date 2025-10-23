"""
Example: Using HalfUNet Decoder with CEBRA

This example demonstrates how to use kirby's HalfUNet as a decoder for CEBRA
embeddings, particularly for reconstructing images from neural activity.
"""

import sys
import torch
import numpy as np

# Add CEBRA to path if needed
# sys.path.append('/path/to/CEBRA')

try:
    from cebra.integrations.decoders import HalfUNetDecoder, halfunet_decoding
    DECODER_AVAILABLE = True
except ImportError:
    DECODER_AVAILABLE = False
    print("Warning: HalfUNet decoder not available. Check kirby installation.")


def example_basic_decoder():
    """Example 1: Basic usage of HalfUNetDecoder class."""
    if not DECODER_AVAILABLE:
        print("Skipping: HalfUNet decoder not available")
        return

    print("=" * 60)
    print("Example 1: Basic HalfUNetDecoder Usage")
    print("=" * 60)

    # Initialize decoder
    decoder = HalfUNetDecoder(
        input_dim=512,
        output_channels=1,
        output_shape=(128, 256),
        latent_dim=512,
    )

    print(f"Decoder initialized:")
    print(f"  - Input dimension: {decoder.input_dim}")
    print(f"  - Output channels: {decoder.output_channels}")
    print(f"  - Output shape: {decoder.output_shape}")
    print(f"  - Latent dimension: {decoder.latent_dim}")

    # Create sample embedding
    batch_size = 8
    embedding = torch.randn(batch_size, 512)

    # Forward pass
    output = decoder(embedding)
    print(f"\nInput shape: {embedding.shape}")
    print(f"Output shape: {output.shape}")

    print("\n")


def example_simple_training():
    """Example 2: Simple training with synthetic data."""
    if not DECODER_AVAILABLE:
        print("Skipping: HalfUNet decoder not available")
        return

    print("=" * 60)
    print("Example 2: Simple Training with Synthetic Data")
    print("=" * 60)

    # Create synthetic data
    print("Creating synthetic data...")
    embedding_train = torch.randn(1000, 256)
    embedding_valid = torch.randn(200, 256)

    # Create target images (simple patterns)
    label_train = torch.randn(1000, 1, 128, 256)
    label_valid = torch.randn(200, 1, 128, 256)

    print(f"Training embeddings: {embedding_train.shape}")
    print(f"Validation embeddings: {embedding_valid.shape}")
    print(f"Training labels: {label_train.shape}")
    print(f"Validation labels: {label_valid.shape}")

    # Train decoder
    print("\nTraining decoder...")
    train_loss, valid_loss, predictions = halfunet_decoding(
        embedding_train=embedding_train,
        embedding_valid=embedding_valid,
        label_train=label_train,
        label_valid=label_valid,
        num_epochs=20,  # Small number for demo
        lr=0.001,
        batch_size=32,
        device='cpu',  # Use 'cuda' if available
        output_channels=1,
        output_shape=(128, 256),
    )

    print(f"\nTraining complete!")
    print(f"Final training loss: {train_loss:.4f}")
    print(f"Final validation loss: {valid_loss:.4f}")
    print(f"Predictions shape: {predictions.shape}")

    print("\n")


def example_rgb_decoding():
    """Example 3: RGB image decoding."""
    if not DECODER_AVAILABLE:
        print("Skipping: HalfUNet decoder not available")
        return

    print("=" * 60)
    print("Example 3: RGB Image Decoding")
    print("=" * 60)

    # Create RGB decoder
    decoder = HalfUNetDecoder(
        input_dim=512,
        output_channels=3,  # RGB
        output_shape=(128, 256),
        latent_dim=1024,
    )

    print(f"RGB Decoder initialized:")
    print(f"  - Output channels: {decoder.output_channels}")

    # Sample forward pass
    embedding = torch.randn(4, 512)
    rgb_output = decoder(embedding)

    print(f"Input shape: {embedding.shape}")
    print(f"RGB output shape: {rgb_output.shape}")
    print(f"Expected: (4, 3, 128, 256) or (4, 128, 256)")

    print("\n")


def example_different_resolutions():
    """Example 4: Different output resolutions."""
    if not DECODER_AVAILABLE:
        print("Skipping: HalfUNet decoder not available")
        return

    print("=" * 60)
    print("Example 4: Different Output Resolutions")
    print("=" * 60)

    resolutions = [
        (64, 128, 256),    # Small latent, small output
        (128, 256, 512),   # Medium
        (256, 512, 1024),  # Large
    ]

    for out_h, out_w, latent_d in resolutions:
        decoder = HalfUNetDecoder(
            input_dim=512,
            output_channels=1,
            output_shape=(out_h, out_w),
            latent_dim=latent_d,
        )

        embedding = torch.randn(2, 512)
        output = decoder(embedding)

        print(f"Resolution: {out_h}x{out_w}, Latent: {latent_d}")
        print(f"  Output shape: {output.shape}")

    print("\n")


def example_loss_functions():
    """Example 5: Different loss functions."""
    if not DECODER_AVAILABLE:
        print("Skipping: HalfUNet decoder not available")
        return

    print("=" * 60)
    print("Example 5: Different Loss Functions")
    print("=" * 60)

    # Create small synthetic dataset
    embedding_train = torch.randn(500, 256)
    embedding_valid = torch.randn(100, 256)
    label_train = torch.randn(500, 1, 128, 256)
    label_valid = torch.randn(100, 1, 128, 256)

    loss_functions = ['mse', 'l1']

    for loss_fn in loss_functions:
        print(f"\nTraining with {loss_fn.upper()} loss...")

        train_loss, valid_loss, _ = halfunet_decoding(
            embedding_train=embedding_train,
            embedding_valid=embedding_valid,
            label_train=label_train,
            label_valid=label_valid,
            num_epochs=10,
            lr=0.001,
            batch_size=32,
            device='cpu',
            loss_fn=loss_fn,
        )

        print(f"  Final validation loss: {valid_loss:.4f}")

    print("\n")


def example_with_cebra_embeddings():
    """Example 6: Complete workflow with CEBRA embeddings."""
    print("=" * 60)
    print("Example 6: Complete CEBRA + HalfUNet Workflow")
    print("=" * 60)

    try:
        import cebra
        CEBRA_AVAILABLE = True
    except ImportError:
        CEBRA_AVAILABLE = False
        print("CEBRA not available, skipping example")
        return

    if not DECODER_AVAILABLE:
        print("HalfUNet decoder not available, skipping example")
        return

    print("Note: This is a conceptual example showing the workflow.")
    print("In practice, you would use real neural data and movie frames.\n")

    # 1. Prepare data
    print("Step 1: Prepare neural data and behavior")
    num_timesteps = 5000
    num_neurons = 100
    behavior_dim = 2

    neural_data = np.random.randn(num_timesteps, num_neurons)
    behavior = np.random.randn(num_timesteps, behavior_dim)

    print(f"  Neural data shape: {neural_data.shape}")
    print(f"  Behavior shape: {behavior.shape}")

    # 2. Train CEBRA model
    print("\nStep 2: Train CEBRA model")
    cebra_model = cebra.CEBRA(
        model_architecture='offset10-model',
        batch_size=512,
        learning_rate=3e-4,
        max_iterations=1000,  # Small for demo
        output_dimension=512,
        verbose=False,
    )

    print("  Training CEBRA... (this may take a moment)")
    cebra_model.fit(neural_data, behavior)
    print("  CEBRA training complete!")

    # 3. Generate embeddings
    print("\nStep 3: Generate embeddings")
    train_indices = slice(0, 4000)
    valid_indices = slice(4000, 5000)

    embedding_train = cebra_model.transform(neural_data[train_indices])
    embedding_valid = cebra_model.transform(neural_data[valid_indices])

    print(f"  Train embeddings shape: {embedding_train.shape}")
    print(f"  Valid embeddings shape: {embedding_valid.shape}")

    # 4. Prepare target images (simulated movie frames)
    print("\nStep 4: Prepare target images")
    movie_frames_train = torch.randn(4000, 1, 128, 256)
    movie_frames_valid = torch.randn(1000, 1, 128, 256)
    print(f"  Train movie frames: {movie_frames_train.shape}")
    print(f"  Valid movie frames: {movie_frames_valid.shape}")

    # 5. Train HalfUNet decoder
    print("\nStep 5: Train HalfUNet decoder")
    train_loss, valid_loss, reconstructed = halfunet_decoding(
        embedding_train=torch.from_numpy(embedding_train).float(),
        embedding_valid=torch.from_numpy(embedding_valid).float(),
        label_train=movie_frames_train,
        label_valid=movie_frames_valid,
        num_epochs=20,
        lr=0.001,
        batch_size=32,
        device='cpu',
        latent_dim=1024,
    )

    print(f"  Training complete!")
    print(f"  Final validation loss: {valid_loss:.4f}")
    print(f"  Reconstructed frames shape: {reconstructed.shape}")

    print("\nWorkflow complete! In practice, you would:")
    print("  - Visualize reconstruction results")
    print("  - Compute reconstruction metrics (SSIM, PSNR, etc.)")
    print("  - Compare reconstructions with originals")

    print("\n")


def example_architecture_comparison():
    """Example 7: Compare different decoder architectures."""
    if not DECODER_AVAILABLE:
        print("Skipping: HalfUNet decoder not available")
        return

    print("=" * 60)
    print("Example 7: Architecture Comparison")
    print("=" * 60)

    print("HalfUNet automatically selects architecture based on latent_dim:\n")

    configs = [
        ("Small UNet", 256, "UNetDecoderOnly"),
        ("Medium UNet", 512, "UNetDecoderOnly"),
        ("Large UNet", 1024, "ComplexLargeUNetDecoderOnly"),
        ("XLarge UNet", 2048, "ComplexLargeUNetDecoderOnly"),
    ]

    for name, latent_dim, expected_arch in configs:
        decoder = HalfUNetDecoder(
            input_dim=512,
            output_channels=1,
            output_shape=(128, 256),
            latent_dim=latent_dim,
        )

        # Count parameters
        total_params = sum(p.numel() for p in decoder.parameters())

        print(f"{name} (latent_dim={latent_dim}):")
        print(f"  Architecture: {expected_arch}")
        print(f"  Total parameters: {total_params:,}")

        # Test forward pass
        embedding = torch.randn(1, 512)
        with torch.no_grad():
            output = decoder(embedding)

        print(f"  Output shape: {output.shape}")
        print()

    print("\n")


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("HalfUNet Decoder Integration with CEBRA - Examples")
    print("=" * 60)
    print("\n")

    if not DECODER_AVAILABLE:
        print("ERROR: HalfUNet decoder is not available.")
        print("Please ensure kirby is installed and in your Python path.")
        print("\nTo fix:")
        print("  import sys")
        print("  sys.path.append('/path/to/CEBRA')")
        sys.exit(1)

    # Run examples
    # Uncomment the examples you want to run

    example_basic_decoder()
    # example_simple_training()
    # example_rgb_decoding()
    # example_different_resolutions()
    # example_loss_functions()
    # example_with_cebra_embeddings()
    # example_architecture_comparison()

    print("Examples completed!")
    print("\nTo run other examples, uncomment them in the main section.")
