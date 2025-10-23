from typing import List, Optional, Tuple, Union

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Import kirby HalfUNet decoder
try:
    from kirby.nn.unet import HalfUNet
    HALFUNET_AVAILABLE = True
except ImportError:
    HALFUNET_AVAILABLE = False
    HalfUNet = None


def ridge_decoding(
    embedding_train: Union[torch.Tensor, dict],
    embedding_valid: Union[torch.Tensor, dict],
    label_train: Union[torch.Tensor, dict],
    label_valid: Union[torch.Tensor, dict],
    n_run: Optional[int] = None,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Perform ridge regression decoding on training and validation embeddings.

    Args:
        embedding_train (Union[torch.Tensor, dict]): Training embeddings.
        embedding_valid (Union[torch.Tensor, dict]): Validation embeddings.
        label_train (Union[torch.Tensor, dict]): Training labels.
        label_valid (Union[torch.Tensor, dict]): Validation labels.
        n_run (Optional[int]): Optional run number for dataset definition.

    Returns:
        Training R2 scores, validation R2 scores, and validation predictions.
    """
    if isinstance(embedding_train, dict):  # only on run 1
        if n_run is None:
            raise ValueError(f"n_run must be specified, got {n_run}.")

        all_train_embeddings = np.concatenate(
            [
                embedding_train[i][n_run].cpu().numpy()
                for i in range(len(embedding_train))
            ],
            axis=0,
        )
        train = np.concatenate(
            [
                label_train[i].continuous.cpu().numpy()
                for i in range(len(label_train))
            ],
            axis=0,
        )
        all_val_embeddings = np.concatenate(
            [
                embedding_valid[i][n_run].cpu().numpy()
                for i in range(len(embedding_valid))
            ],
            axis=0,
        )
        valid = np.concatenate(
            [
                label_valid[i].continuous.cpu().numpy()
                for i in range(len(label_valid))
            ],
            axis=0,
        )
    else:
        all_train_embeddings = embedding_train.cpu().numpy()
        train = label_train.cpu().numpy()
        all_val_embeddings = embedding_valid.cpu().numpy()
        valid = label_valid.cpu().numpy()

    decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 0, 9)})
    decoder.fit(all_train_embeddings, train)

    train_prediction = decoder.predict(all_train_embeddings)
    train_scores = sklearn.metrics.r2_score(train,
                                            train_prediction,
                                            multioutput="raw_values").tolist()
    valid_prediction = decoder.predict(all_val_embeddings)
    valid_scores = sklearn.metrics.r2_score(valid,
                                            valid_prediction,
                                            multioutput="raw_values").tolist()

    return train_scores, valid_scores, valid_prediction


class SingleLayerDecoder(nn.Module):
    """Supervised module to predict behaviors.

    Note:
        By default, the output dimension is 2, to predict x/y velocity
        (Perich et al., 2018).
    """

    def __init__(self, input_dim, output_dim=2):
        super(SingleLayerDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TwoLayersDecoder(nn.Module):
    """Supervised module to predict behaviors.

    Note:
        By default, the output dimension is 2, to predict x/y velocity
        (Perich et al., 2018).
    """

    def __init__(self, input_dim, output_dim=2):
        super(TwoLayersDecoder, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(),
                                nn.Linear(32, output_dim))

    def forward(self, x):
        return self.fc(x)


class HalfUNetDecoder(nn.Module):
    """UNet-based decoder for reconstructing high-dimensional outputs from embeddings.

    This decoder uses the HalfUNet architecture from kirby.nn.unet to decode
    embeddings into images or other 2D structured outputs. It's particularly
    useful for decoding neural embeddings back to visual stimuli or other
    spatially structured data.

    Args:
        input_dim: Dimension of the input embedding
        output_channels: Number of output channels (default: 1 for grayscale images)
        output_shape: Tuple of (height, width) for the output image size.
            Default is (128, 256).
        latent_dim: Internal latent dimension of the UNet. If None, uses input_dim.
            For best results with large embeddings (>=1024), the ComplexLargeUNetDecoderOnly
            will be used automatically.

    Example:
        >>> decoder = HalfUNetDecoder(input_dim=512, output_channels=1, output_shape=(128, 256))
        >>> embedding = torch.randn(32, 512)  # batch_size=32, embedding_dim=512
        >>> output = decoder(embedding)  # shape: (32, 128, 256) for single channel
        >>> # or (32, C, 128, 256) for multi-channel outputs
    """

    def __init__(
        self,
        input_dim: int,
        output_channels: int = 1,
        output_shape: Tuple[int, int] = (128, 256),
        latent_dim: Optional[int] = None,
    ):
        super(HalfUNetDecoder, self).__init__()

        if not HALFUNET_AVAILABLE:
            raise ImportError(
                "HalfUNet is not available. Please ensure kirby is installed "
                "and accessible in your Python environment."
            )

        self.input_dim = input_dim
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.latent_dim = latent_dim if latent_dim is not None else input_dim

        # Project input embedding to the latent dimension if needed
        if self.input_dim != self.latent_dim:
            self.projection = nn.Linear(input_dim, self.latent_dim)
        else:
            self.projection = nn.Identity()

        # Initialize HalfUNet decoder
        self.unet = HalfUNet(
            in_channels=1,
            out_channels=output_channels,
            latent_dim=self.latent_dim
        )

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_channels, height, width)
            or (batch_size, height, width) if output_channels=1 and the UNet
            squeezes the channel dimension.
        """
        # Project to latent dimension
        x = self.projection(x)

        # Pass through UNet decoder
        output = self.unet(x)

        # Ensure output has the correct shape
        if output.dim() == 3:
            # UNet squeezed the channel dimension, add it back if needed
            if self.output_channels > 1:
                raise ValueError(
                    f"Expected output with {self.output_channels} channels, "
                    f"but got shape {output.shape}"
                )
            # Keep as (batch, H, W) for single channel
            return output
        else:
            # Output is (batch, C, H, W)
            return output


def mlp_decoding(
    embedding_train: Union[dict, torch.Tensor],
    embedding_valid: Union[dict, torch.Tensor],
    label_train: Union[dict, torch.Tensor],
    label_valid: Union[dict, torch.Tensor],
    num_epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 500,
    device: str = "cuda",
    model_type: str = "SingleLayerMLP",
    n_run: Optional[int] = None,
):
    """ Perform MLP decoding on training and validation embeddings.

    Args:
        embedding_train (Union[dict, torch.Tensor]): Training embeddings.
        embedding_valid (Union[dict, torch.Tensor]): Validation embeddings.
        label_train (Union[dict, torch.Tensor]): Training labels.
        label_valid (Union[dict, torch.Tensor]): Validation labels.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        device (str): Device to run the model on ('cuda' or 'cpu').
        model_type (str): Type of MLP model to use ('SingleLayerMLP' or 'TwoLayersMLP').
        n_run (Optional[int]): Optional run number for dataset definition.

    Returns:
        Training R2 scores, validation R2 scores, and validation predictions.
    """
    if len(label_train.shape) == 1:
        label_train = label_train[:, None]
        label_valid = label_valid[:, None]

    if isinstance(embedding_train, dict):  # only on run 1
        if n_run is None:
            raise ValueError(f"n_run must be specified, got {n_run}.")

        all_train_embeddings = torch.cat(
            [embedding_train[i][n_run] for i in range(len(embedding_train))],
            axis=0)
        train = torch.cat(
            [label_train[i].continuous for i in range(len(label_train))],
            axis=0)
        all_val_embeddings = torch.cat(
            [embedding_valid[i][n_run] for i in range(len(embedding_valid))],
            axis=0)
        valid = torch.cat(
            [label_valid[i].continuous for i in range(len(label_valid))],
            axis=0)
    else:
        all_train_embeddings = embedding_train
        train = label_train
        all_val_embeddings = embedding_valid
        valid = label_valid

    dataset = TensorDataset(all_train_embeddings.to(device), train.to(device))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = all_train_embeddings.shape[1]
    output_dim = train.shape[1]
    if model_type == "SingleLayerMLP":
        model = SingleLayerDecoder(input_dim=input_dim, output_dim=output_dim)
    elif model_type == "TwoLayersMLP":
        model = TwoLayersDecoder(input_dim=input_dim, output_dim=output_dim)
    else:
        raise NotImplementedError()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    train_pred = model(all_train_embeddings.to(device))
    train_r2 = sklearn.metrics.r2_score(
        y_true=train.cpu().numpy(),
        y_pred=train_pred.cpu().detach().numpy(),
        multioutput="raw_values",
    ).tolist()

    valid_pred = model(all_val_embeddings.to(device))
    valid_r2 = sklearn.metrics.r2_score(
        y_true=valid.cpu().numpy(),
        y_pred=valid_pred.cpu().detach().numpy(),
        multioutput="raw_values",
    ).tolist()

    return train_r2, valid_r2, valid_pred


def halfunet_decoding(
    embedding_train: Union[dict, torch.Tensor],
    embedding_valid: Union[dict, torch.Tensor],
    label_train: Union[dict, torch.Tensor],
    label_valid: Union[dict, torch.Tensor],
    num_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cuda",
    output_channels: int = 1,
    output_shape: Tuple[int, int] = (128, 256),
    latent_dim: Optional[int] = None,
    n_run: Optional[int] = None,
    loss_fn: str = "mse",
):
    """Perform HalfUNet decoding on training and validation embeddings.

    This function trains a HalfUNet decoder to reconstruct images or 2D structured
    outputs from embeddings. It's particularly useful for decoding neural activity
    back to visual stimuli.

    Args:
        embedding_train: Training embeddings, shape (N, embedding_dim)
        embedding_valid: Validation embeddings, shape (M, embedding_dim)
        label_train: Training labels (target images), shape (N, C, H, W) or (N, H, W)
        label_valid: Validation labels (target images), shape (M, C, H, W) or (M, H, W)
        num_epochs: Number of training epochs (default: 100)
        lr: Learning rate for the optimizer (default: 0.001)
        batch_size: Batch size for training (default: 32)
        device: Device to run the model on ('cuda' or 'cpu')
        output_channels: Number of output channels (default: 1 for grayscale)
        output_shape: Tuple of (height, width) for output images (default: (128, 256))
        latent_dim: Internal latent dimension. If None, uses embedding dimension
        n_run: Optional run number for multi-session datasets
        loss_fn: Loss function to use ('mse', 'l1', or 'bce'). Default: 'mse'

    Returns:
        tuple: (train_loss, valid_loss, valid_predictions)
            - train_loss: Final training loss
            - valid_loss: Final validation loss
            - valid_predictions: Reconstructed images on validation set

    Example:
        >>> # Embeddings from CEBRA
        >>> embedding_train = torch.randn(1000, 512)
        >>> embedding_valid = torch.randn(200, 512)
        >>> # Target images (e.g., visual stimuli)
        >>> label_train = torch.randn(1000, 1, 128, 256)
        >>> label_valid = torch.randn(200, 1, 128, 256)
        >>> train_loss, valid_loss, predictions = halfunet_decoding(
        ...     embedding_train, embedding_valid,
        ...     label_train, label_valid,
        ...     num_epochs=100, device='cuda'
        ... )
    """
    if not HALFUNET_AVAILABLE:
        raise ImportError(
            "HalfUNet is not available. Please ensure kirby is installed."
        )

    # Handle multi-session datasets
    if isinstance(embedding_train, dict):
        if n_run is None:
            raise ValueError("n_run must be specified for multi-session datasets")

        all_train_embeddings = torch.cat(
            [embedding_train[i][n_run] for i in range(len(embedding_train))],
            dim=0
        )
        all_val_embeddings = torch.cat(
            [embedding_valid[i][n_run] for i in range(len(embedding_valid))],
            dim=0
        )

        # Handle labels - assuming they're stored similarly
        if isinstance(label_train, dict):
            train_labels = torch.cat(
                [label_train[i] for i in range(len(label_train))],
                dim=0
            )
            valid_labels = torch.cat(
                [label_valid[i] for i in range(len(label_valid))],
                dim=0
            )
        else:
            train_labels = label_train
            valid_labels = label_valid
    else:
        all_train_embeddings = embedding_train
        all_val_embeddings = embedding_valid
        train_labels = label_train
        valid_labels = label_valid

    # Ensure labels have the right shape (batch, C, H, W)
    if train_labels.dim() == 3:
        train_labels = train_labels.unsqueeze(1)  # Add channel dimension
    if valid_labels.dim() == 3:
        valid_labels = valid_labels.unsqueeze(1)

    # Create dataloaders
    train_dataset = TensorDataset(
        all_train_embeddings.to(device),
        train_labels.to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = all_train_embeddings.shape[1]
    model = HalfUNetDecoder(
        input_dim=input_dim,
        output_channels=output_channels,
        output_shape=output_shape,
        latent_dim=latent_dim,
    )
    model.to(device)

    # Loss function
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "l1":
        criterion = nn.L1Loss()
    elif loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for embeddings, targets in train_loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)

            # Ensure output matches target shape
            if outputs.dim() == 3 and targets.dim() == 4:
                outputs = outputs.unsqueeze(1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Train predictions
        train_pred = model(all_train_embeddings.to(device))
        if train_pred.dim() == 3 and train_labels.dim() == 4:
            train_pred = train_pred.unsqueeze(1)
        train_loss = criterion(train_pred, train_labels.to(device)).item()

        # Validation predictions
        valid_pred = model(all_val_embeddings.to(device))
        if valid_pred.dim() == 3 and valid_labels.dim() == 4:
            valid_pred = valid_pred.unsqueeze(1)
        valid_loss = criterion(valid_pred, valid_labels.to(device)).item()

    return train_loss, valid_loss, valid_pred.cpu()
