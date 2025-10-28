import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchmetrics import R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from kirby.taxonomy import OutputType
import numpy as np


class FFTLoss(nn.Module):
    """
    Frequency Domain (FFT) Loss.
    
    Computes the loss in the frequency domain between the predicted and target images.
    It encourages the model to match the high-frequency details of the target image.
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'mean'):
        """
        Args:
            loss_type (str): The type of loss to use ('l1' or 'l2'). Default is 'l1'.
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(FFTLoss, self).__init__()

        if loss_type.lower() == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type.lower() == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'l1' or 'l2'.")
            
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): The predicted image tensor, shape [B, C, H, W].
            true (torch.Tensor): The ground truth image tensor, shape [B, C, H, W].

        Returns:
            torch.Tensor: The calculated FFT loss as a scalar tensor.
        """
        # --- Defensive checks ---
        if pred.shape != true.shape:
            raise ValueError(f"Input shapes must be the same. Got {pred.shape} and {true.shape}")

        # Clamp inputs to valid range
        pred = torch.clamp(pred, -10.0, 10.0)
        true = torch.clamp(true, -10.0, 10.0)

        # --- Compute FFT ---
        # Apply N-dimensional FFT. Using `fftn` is general for 2D/3D.
        # The result is a complex tensor.
        pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
        true_fft = torch.fft.fftn(true, dim=(-2, -1))

        # --- Compute loss on the magnitude of the spectrum ---
        # The phase information is usually discarded. We compute loss on the absolute values (magnitudes).
        pred_fft_mag = torch.abs(pred_fft)
        true_fft_mag = torch.abs(true_fft)

        # Clamp FFT magnitudes to prevent extreme values
        pred_fft_mag = torch.clamp(pred_fft_mag, 0.0, 1e4)
        true_fft_mag = torch.clamp(true_fft_mag, 0.0, 1e4)

        # Calculate the loss between the frequency spectrums
        loss = self.loss_fn(pred_fft_mag, true_fft_mag, reduction=self.reduction)

        # Clamp final loss
        loss = torch.clamp(loss, 0.0, 1e3)

        return loss

class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss (GDL)
    
    Computes the L1 loss between the gradients of the predicted and target images.
    The gradients are computed using Sobel filters.
    """
    def __init__(self, channels: int, loss_type: str = 'l1'):
        """
        Args:
            channels (int): The number of channels in the input images (e.g., 3 for RGB).
            loss_type (str): The type of loss to use ('l1' or 'l2'). Default is 'l1'.
        """
        super(GradientDifferenceLoss, self).__init__()
        self.channels = channels
        
        # Define Sobel filters for x and y gradients
        kernel_x = torch.from_numpy(np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])).float().unsqueeze(0).unsqueeze(0)

        kernel_y = torch.from_numpy(np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])).float().unsqueeze(0).unsqueeze(0)

        # Expand kernels to match the number of input channels
        kernel_x = kernel_x.repeat(self.channels, 1, 1, 1)
        kernel_y = kernel_y.repeat(self.channels, 1, 1, 1)

        # Create convolutional layers for applying the Sobel filters
        # `groups=channels` applies the filter independently to each channel
        self.conv_x = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
                                kernel_size=3, stride=1, padding=1, bias=False, 
                                groups=self.channels)
        
        self.conv_y = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
                                kernel_size=3, stride=1, padding=1, bias=False, 
                                groups=self.channels)

        # Set the weights of the convolutional layers to the Sobel kernels
        self.conv_x.weight.data = kernel_x
        self.conv_y.weight.data = kernel_y
        
        # Freeze the weights
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

        # Define the loss function
        if loss_type.lower() == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type.lower() == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'l1' or 'l2'.")

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): The predicted image tensor, shape [B, C, H, W].
            true (torch.Tensor): The ground truth image tensor, shape [B, C, H, W].

        Returns:
            torch.Tensor: The calculated gradient difference loss as a scalar tensor.
        """
        # Ensure the conv layers are on the same device as the input tensors
        self.conv_x.to(pred.device)
        self.conv_y.to(pred.device)

        # Clamp inputs to valid range
        pred = torch.clamp(pred, -10.0, 10.0)
        true = torch.clamp(true, -10.0, 10.0)

        # Calculate gradients in x and y directions for both images
        pred_grad_x = self.conv_x(pred)
        true_grad_x = self.conv_x(true)

        pred_grad_y = self.conv_y(pred)
        true_grad_y = self.conv_y(true)

        # Compute the loss for each direction
        loss_x = self.loss_fn(pred_grad_x, true_grad_x)
        loss_y = self.loss_fn(pred_grad_y, true_grad_y)

        # Total loss is the sum of the losses in both directions
        total_loss = loss_x + loss_y

        # Clamp to prevent extreme values
        total_loss = torch.clamp(total_loss, 0.0, 1e3)

        return total_loss

class LaplacianLoss(nn.Module):
    """
    Laplacian (edge) loss between predicted and target images.
    Applies a Laplacian filter to both images and computes L1 or L2 distance between the filtered outputs.
    
    Args:
        loss_type (str): 'l1' or 'l2'. Specifies L1 or L2 on Laplacian responses.
        reduction (str): 'mean' or 'sum' or 'none' for reduction over batch.
        normalize (bool): If True, normalize per-sample Laplacian maps by their max abs value to reduce scale sensitivity.
        eps (float): Small epsilon for numerical stability in normalization.
        multi_scale (bool): If True, compute Laplacian loss on an image pyramid (scales factor 1, 1/2, 1/4) and average.
    """
    def __init__(self, loss_type='l1', reduction='mean', normalize=False, eps=1e-6, multi_scale=False):
        super().__init__()
        assert loss_type in ('l1', 'l2'), "loss_type must be 'l1' or 'l2'"
        assert reduction in ('mean', 'sum', 'none'), "reduction must be 'mean','sum' or 'none'"
        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize = normalize
        self.eps = eps
        self.multi_scale = multi_scale
        
        # 3x3 discrete Laplacian kernel (common choice)
        # [[0,  1, 0],
        #  [1, -4, 1],
        #  [0,  1, 0]]
        lap = torch.tensor([[0.0,  1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0,  1.0, 0.0]], dtype=torch.float32)
        self.register_buffer('kernel_3x3', lap.unsqueeze(0).unsqueeze(0))  # shape (1,1,3,3)
    
    def _apply_lap(self, x):
        """
        x: tensor (N, C, H, W)
        returns: tensor (N, C, H, W) of Laplacian response
        """
        N, C, H, W = x.shape
        # expand kernel to per-channel groups conv: shape (C,1,k,k)
        kernel = self.kernel_3x3.repeat(C, 1, 1, 1).to(x.device)
        # padding=1 for same spatial size
        # use groups=C to apply same kernel independently per channel
        lap = F.conv2d(x, kernel, bias=None, stride=1, padding=1, groups=C)
        return lap

    def _single_scale_loss(self, pred, target):
        lap_pred = self._apply_lap(pred)
        lap_target = self._apply_lap(target)
        
        if self.normalize:
            # normalize per-sample (per N) by max abs value in target Lap response to reduce scale sensitivity
            # shape: (N,1,1,1)
            max_vals = lap_target.abs().reshape(lap_target.shape[0], -1).max(dim=1)[0].clamp(min=self.eps)
            max_vals = max_vals.view(-1, 1, 1, 1)
            lap_pred = lap_pred / max_vals
            lap_target = lap_target / max_vals
        
        if self.loss_type == 'l1':
            loss_map = (lap_pred - lap_target).abs()
        else:
            loss_map = (lap_pred - lap_target) ** 2
        
        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        else:
            return loss_map  # no reduction
    
    def forward(self, pred, target):
        """
        pred, target: tensors (N, C, H, W), expected float, same device.
        """
        assert pred.shape == target.shape, "pred and target must have same shape"
        if not self.multi_scale:
            return self._single_scale_loss(pred, target)
        else:
            # multi-scale: factors 1.0, 0.5, 0.25 (if image large enough)
            scales = [1.0, 0.5, 0.25]
            losses = []
            for s in scales:
                if s == 1.0:
                    p = pred
                    t = target
                else:
                    # use area interpolation for downsampling to keep aliasing low
                    h = max(1, int(round(pred.shape[2] * s)))
                    w = max(1, int(round(pred.shape[3] * s)))
                    p = F.interpolate(pred, size=(h, w), mode='area')
                    t = F.interpolate(target, size=(h, w), mode='area')
                losses.append(self._single_scale_loss(p, t))
            # average scales
            total = sum(losses) / len(losses)
            return total



class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()
        
        # VGG19 모델을 불러와서 중간 레이어만 사용
        vgg = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.eval().to("cpu")  # VGG19 모델 로드 (pretrained)
        
        # 우리가 사용할 레이어를 지정 (보통 2~5번째 레이어 사용)
        if layers is None:
            layers = [0, 5, 10, 19, 28]  # 기본 레이어 설정 (예: VGG19의 중간 레이어)

        # 지정된 레이어만 사용할 수 있도록 설정
        self.vgg = nn.Sequential(*[vgg[i] for i in layers])
        
        # ImageNet 정규화 파라미터
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 파라미터 업데이트 금지 (VGG19는 frozen 모델)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred_image, target_image):
        """
        pred_image: 복원된 이미지 (prediction)
        target_image: 실제 이미지 (ground truth)
        
        perceptual loss는 주로 MSE를 사용하여 feature space에서 차이 계산
        """
        # VGG19가 사용되는 device에 맞춰서 pred_image와 target_image도 해당 device로 옮겨줍니다.
        device = next(self.vgg.parameters()).device  # VGG19의 device를 확인
        pred_image = pred_image.detach().to("cpu").float() #.to(device).float()  # pred_image를 VGG19와 동일한 device로 이동
        target_image = target_image.detach().to("cpu").float() #.to(device).float()  # target_image를 VGG19와 동일한 device로 이동

        pred_image = (pred_image - self.mean) / self.std
        target_image = (target_image - self.mean) / self.std
        pred_features = self.vgg(pred_image)  # 예측 이미지의 feature 추출
        target_features = self.vgg(target_image)  # 실제 이미지의 feature 추출
        
        # MSE Loss를 사용하여 feature 차이를 계산
        loss = F.mse_loss(pred_features, target_features)
        return loss


class AlexNetPerceptualLoss(nn.Module):
    def __init__(self, layer=3):
        super().__init__()
        alexnet = models.alexnet(weights='IMAGENET1K_V1').features.eval()
        self.features = nn.Sequential(*list(alexnet.children())[:layer+1])

        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet 정규화 파라미터
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        device = next(self.features.parameters()).device

        # Convert grayscale to RGB if needed (repeat channel 3 times)
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        pred = pred.to(device)
        target = target.to(device)

        # Normalize
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Extract features and compute loss
        pred_features = self.features(pred)
        target_features = self.features(target)

        loss = F.mse_loss(pred_features, target_features)

        # Clamp loss to prevent infinity
        loss = torch.clamp(loss, max=1e6)

        return loss


class TVLoss(nn.Module):
    """Total Variation Loss for image smoothness"""
    
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            TV loss value
        """
        batch_size, channels, height, width = x.size()
        
        # Calculate differences along height and width dimensions
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # Vertical differences
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # Horizontal differences
        
        # Sum all differences
        tv_loss = torch.sum(diff_h) + torch.sum(diff_w)
        
        # Normalize by the number of pixels
        tv_loss = tv_loss / (batch_size * channels * height * width)
        
        return self.weight * tv_loss
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        """
        Contrastive loss function based on cosine similarity.
        Args:
            temperature (float): Scaling factor for the similarity scores.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, f_masked, f_original):
        """
        Compute contrastive loss.
        
        Args:
            f_masked (Tensor): Features from masked sequence (batch, N_dim, dim).
            f_original (Tensor): Features from original sequence (batch, N_dim, dim).
        
        Returns:
            loss (Tensor): Contrastive loss value.
        """
        # Normalize embeddings along the last dimension (dim)
        f_masked = F.normalize(f_masked, dim=-1)
        f_original = F.normalize(f_original, dim=-1)

        # Compute cosine similarity along the last dimension (dim)
        similarity = torch.sum(f_masked * f_original, dim=-1)  # Shape: (batch, N_dim)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # Contrastive loss (maximize similarity across all N_dim)
        loss = torch.mean(1 - similarity)
        
        return loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, output, target):
        device = output.get_device() if output.get_device() >= 0 else torch.device('cpu')
        self.ssim = self.ssim.to(device)

        # Clamp values to valid range [0, 1]
        output = torch.clamp(output, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        ssim_value = self.ssim(output, target)

        # Clamp SSIM result to prevent NaN
        ssim_value = torch.clamp(ssim_value, -1.0, 1.0)

        loss = 1 - ssim_value

        # Ensure loss is non-negative and bounded
        loss = torch.clamp(loss, 0.0, 2.0)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, loss_type='l1'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_type = loss_type

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        if self.loss_type == 'l1':
            base_loss = F.l1_loss(pred, target, reduction='none')

        # Calculate confidence (1 - normalized error)
        error = torch.abs(pred - target)
        max_error = error.max()

        # Add larger epsilon for numerical stability
        max_error = torch.clamp(max_error, min=1e-6)

        confidence = 1 - (error / (max_error + 1e-6))
        confidence = torch.clamp(confidence, 0.0, 1.0)

        # Focal weight: (1-confidence)^gamma
        focal_weight = (1 - confidence) ** self.gamma

        # Clamp focal weight to prevent extreme values
        focal_weight = torch.clamp(focal_weight, 0.0, 10.0)

        # Apply focal loss
        focal_loss = self.alpha * focal_weight * base_loss

        # Clamp final loss
        focal_loss = torch.clamp(focal_loss, 0.0, 1e3)

        return focal_loss.mean()

class MultiScaleLoss(nn.Module):
    def __init__(self, loss_type='l1', scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        """
        Multi-scale loss function
        
        Args:
            loss_type (str): Type of loss ('l1', 'l2', 'smooth_l1')
            scales (list): Scale factors for downsampling [1.0, 0.5, 0.25]
            weights (list): Weights for each scale [1.0, 0.5, 0.25]
        """
        super(MultiScaleLoss, self).__init__()
        
        assert len(scales) == len(weights), "scales and weights must have same length"
        
        self.scales = scales
        self.weights = weights
        self.loss_type = loss_type
        
        # Define loss function
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred, target):
        """
        Calculate multi-scale loss
        
        Args:
            pred (torch.Tensor): Predicted image [B, C, H, W]
            target (torch.Tensor): Target image [B, C, H, W]
            
        Returns:
            torch.Tensor: Multi-scale loss value
        """
        total_loss = 0.0
        pred = pred.float()
        target = target.float()
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1.0:
                # Original resolution
                scaled_pred = pred
                scaled_target = target
            else:
                # Downsample both pred and target
                scaled_pred = F.interpolate(pred, scale_factor=scale, 
                                          mode='bilinear', align_corners=False)
                scaled_target = F.interpolate(target, scale_factor=scale, 
                                            mode='bilinear', align_corners=False)
            
            # Calculate loss at this scale
            scale_loss = self.loss_fn(scaled_pred, scaled_target)
            total_loss += weight * scale_loss
        
        return total_loss

def compute_loss_or_metric(
    loss_or_metric: str,
    output_type: OutputType,
    output: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    decoder_id: str,
) -> torch.Tensor:
    r"""Helper function to compute various losses or metrics for a given output type.

    It supports both continuous and discrete output types, and a variety of losses
    and metrics, including mse loss, binary cross entropy loss, and R2 score.

    Args:
        loss_or_metric: The name of the metric to compute. e.g. bce, mse
        output_type: The nature of the output. One of the values from OutputType. e.g. MULTINOMIAL
        output: The output tensor.
        target: The target tensor.
        weights: The sample-wise weights for the loss computation.
    """
    if 'NATURAL_' in decoder_id and output_type == None:
        # [batch, 3, 32, 64]
        ssim = StructuralSimilarityIndexMeasure()
        return ssim(output, target)
    else:        
        if output_type == OutputType.CONTINUOUS:
            if loss_or_metric == "mse":
                # TODO mse could be used as a loss or as a metric. Currently it fails when
                # called as a metric
                # MSE loss
                loss_noreduce = F.mse_loss(output, target, reduction="none").mean(dim=1)
                return (weights * loss_noreduce).sum() / weights.sum()
            elif loss_or_metric == "r2":
                r2score = R2Score(num_outputs=target.shape[1])
                return r2score(output, target)
            elif loss_or_metric == "frame_diff_acc":
                normalized_window = 30 / 450
                differences = torch.abs(output - target)
                correct_predictions = differences <= normalized_window
                accuracy = (
                    correct_predictions.float().mean()
                )  # Convert boolean tensor to float and calculate mean
                return accuracy
            else:
                raise NotImplementedError(
                    f"Loss/Metric {loss_or_metric} not implemented for continuous output"
                )

        if output_type in [
            OutputType.BINARY,
            OutputType.MULTINOMIAL,
            OutputType.MULTILABEL,
        ]:
            if decoder_id == 'CRE_LINE':
                unique_classes = torch.unique(target)
                class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
                device = target.get_device()
                target = torch.tensor([class_to_index[cls.item()] for cls in target])
                target = target.to(device)

            if loss_or_metric == "bce":
                target = target.to(torch.long).squeeze()
                # target = target.squeeze(dim=1)
                loss_noreduce = F.cross_entropy(output, target , reduction="none")
                if loss_noreduce.ndim > 1:
                    loss_noreduce = loss_noreduce.mean(dim=1)
                return (weights * loss_noreduce).sum() / weights.sum()
            elif loss_or_metric == "mallows_distance":
                num_classes = output.size(-1)
                output = torch.softmax(output, dim=-1).view(-1, num_classes)
                target = target.view(-1, 1)
                weights = weights.view(-1)
                # Mallow distance
                target = torch.zeros_like(output).scatter_(1, target, 1.0)
                # we compute the mallow distance as the sum of the squared differences
                loss = torch.mean(
                    torch.square(
                        torch.cumsum(target, dim=-1) - torch.cumsum(output, dim=-1)
                    ),
                    dim=-1,
                )
                loss = (weights * loss).sum() / weights.sum()
                return loss
            # elif loss_or_metric == "accuracy":
            #     pred_class = torch.argmax(output, dim=1)
            #     return (pred_class == target.squeeze()).sum() / len(target)

            elif loss_or_metric == "accuracy":
                pred_class = torch.argmax(output, dim=1)
                
                # --- 수정 전 (Before) ---
                # return (pred_class == target.squeeze()).sum() / len(target)

                # --- 수정 후 (After) ---
                # 1. GPU에 있는 '맞춘 개수' 텐서
                correct_predictions = (pred_class == target.squeeze()).sum()

                # 2. Python 숫자인 '전체 개수'를 GPU 텐서로 명시적으로 변환
                total_count = torch.tensor(len(target), device=correct_predictions.device)

                # 3. GPU 텐서끼리의 나눗셈을 수행하여, 결과가 반드시 GPU에 있도록 보장
                return correct_predictions / total_count

            elif loss_or_metric == "frame_diff_acc":
                pred_class = torch.argmax(output, dim=1)
                difference = torch.abs(pred_class - target.squeeze())
                correct_predictions = difference <= 30
                return correct_predictions.float().mean()
            else:
                raise NotImplementedError(
                    f"Loss/Metric {loss_or_metric} not implemented for binary/multilabel "
                    "output"
                )

        raise NotImplementedError(
            "I don't know how to handle this task type. Implement plis"
        )
