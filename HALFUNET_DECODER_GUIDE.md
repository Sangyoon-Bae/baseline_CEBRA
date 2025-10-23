# HalfUNet Decoder Integration Guide

이 가이드는 CEBRA에서 kirby의 HalfUNet을 디코더로 사용하는 방법을 설명합니다.

## 개요

HalfUNet은 kirby.nn.unet 모듈의 강력한 디코더 아키텍처로, CEBRA 임베딩을 이미지나 2D 구조화된 출력으로 재구성하는 데 최적화되어 있습니다. 이 통합을 통해 신경 활동 임베딩을 시각적 자극이나 다른 공간적으로 구조화된 데이터로 디코딩할 수 있습니다.

## 주요 기능

- **고품질 재구성**: RRDB(Residual-in-Residual Dense Block) 기반의 복잡한 아키텍처
- **유연한 해상도**: 다양한 출력 크기 지원 (기본: 128x256)
- **자동 아키텍처 선택**: 임베딩 크기에 따라 최적의 디코더 자동 선택
  - latent_dim < 1024: 기본 UNetDecoderOnly 사용
  - latent_dim >= 1024: ComplexLargeUNetDecoderOnly 사용 (더 고품질)
- **멀티채널 지원**: 그레이스케일(1채널) 또는 RGB(3채널) 등 다양한 출력 형식

## 설치 요구사항

```bash
# kirby가 Python 경로에 있는지 확인
import sys
sys.path.append('/path/to/CEBRA')  # kirby 폴더가 있는 경로

# 필요한 패키지
import torch
from cebra.integrations.decoders import HalfUNetDecoder, halfunet_decoding
```

## 사용법

### 1. 기본 사용 - HalfUNetDecoder 클래스

```python
import torch
from cebra.integrations.decoders import HalfUNetDecoder

# 디코더 초기화
decoder = HalfUNetDecoder(
    input_dim=512,           # CEBRA 임베딩 차원
    output_channels=1,       # 출력 채널 수 (1=그레이스케일, 3=RGB)
    output_shape=(128, 256), # 출력 이미지 크기
    latent_dim=512,          # 내부 latent 차원 (None이면 input_dim 사용)
)

# 추론
embedding = torch.randn(32, 512)  # (batch_size, embedding_dim)
reconstructed = decoder(embedding)  # (32, 128, 256) 또는 (32, 1, 128, 256)
```

### 2. 전체 학습 파이프라인 - halfunet_decoding 함수

```python
import torch
from cebra.integrations.decoders import halfunet_decoding

# CEBRA로 얻은 임베딩
embedding_train = torch.randn(1000, 512)
embedding_valid = torch.randn(200, 512)

# 타겟 이미지 (예: 시각적 자극)
label_train = torch.randn(1000, 1, 128, 256)  # (N, C, H, W)
label_valid = torch.randn(200, 1, 128, 256)

# 디코더 학습
train_loss, valid_loss, predictions = halfunet_decoding(
    embedding_train=embedding_train,
    embedding_valid=embedding_valid,
    label_train=label_train,
    label_valid=label_valid,
    num_epochs=100,
    lr=0.001,
    batch_size=32,
    device='cuda',
    output_channels=1,
    output_shape=(128, 256),
    loss_fn='mse',  # 'mse', 'l1', 또는 'bce'
)

print(f"Final validation loss: {valid_loss:.4f}")
print(f"Predictions shape: {predictions.shape}")
```

### 3. CEBRA와 통합된 전체 워크플로우

```python
import torch
import cebra
from cebra.integrations.decoders import halfunet_decoding

# 1. CEBRA 모델 학습
cebra_model = cebra.CEBRA(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=3e-4,
    max_iterations=10000,
    output_dimension=512,
)

# 신경 데이터와 연속적인 행동 변수
neural_data = torch.randn(10000, 100)  # (시간, 뉴런)
behavior = torch.randn(10000, 2)       # (시간, 행동차원)

cebra_model.fit(neural_data, behavior)

# 2. 임베딩 생성
train_indices = slice(0, 8000)
valid_indices = slice(8000, 10000)

embedding_train = cebra_model.transform(neural_data[train_indices])
embedding_valid = cebra_model.transform(neural_data[valid_indices])

# 3. 타겟 시각적 자극 준비 (예: 영화 프레임)
movie_frames_train = load_movie_frames(train_indices)  # (8000, 1, 128, 256)
movie_frames_valid = load_movie_frames(valid_indices)  # (2000, 1, 128, 256)

# 4. HalfUNet 디코더 학습
train_loss, valid_loss, reconstructed_frames = halfunet_decoding(
    embedding_train=torch.from_numpy(embedding_train),
    embedding_valid=torch.from_numpy(embedding_valid),
    label_train=movie_frames_train,
    label_valid=movie_frames_valid,
    num_epochs=100,
    lr=0.001,
    batch_size=32,
    device='cuda',
    latent_dim=1024,  # 더 큰 latent dimension으로 고품질 재구성
)

# 5. 결과 시각화
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # 원본
    axes[0, i].imshow(movie_frames_valid[i, 0], cmap='gray')
    axes[0, i].set_title(f'Original {i}')
    axes[0, i].axis('off')

    # 재구성
    axes[1, i].imshow(reconstructed_frames[i, 0].detach().numpy(), cmap='gray')
    axes[1, i].set_title(f'Reconstructed {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('reconstruction_results.png')
```

## 고급 사용 예제

### 1. RGB 이미지 디코딩

```python
# RGB 출력을 위한 설정
decoder = HalfUNetDecoder(
    input_dim=512,
    output_channels=3,      # RGB
    output_shape=(128, 256),
    latent_dim=1024,        # 더 복잡한 출력을 위해 큰 latent
)

# 학습
train_loss, valid_loss, rgb_predictions = halfunet_decoding(
    embedding_train=embedding_train,
    embedding_valid=embedding_valid,
    label_train=rgb_images_train,  # (N, 3, 128, 256)
    label_valid=rgb_images_valid,  # (M, 3, 128, 256)
    output_channels=3,
    num_epochs=150,
    device='cuda',
)
```

### 2. 다양한 해상도

```python
# 더 큰 출력 해상도
decoder_large = HalfUNetDecoder(
    input_dim=512,
    output_channels=1,
    output_shape=(256, 512),  # 더 큰 이미지
    latent_dim=2048,          # 복잡한 출력을 위해 더 큰 latent
)

# 작은 출력 해상도
decoder_small = HalfUNetDecoder(
    input_dim=512,
    output_channels=1,
    output_shape=(64, 128),   # 작은 이미지
    latent_dim=256,
)
```

### 3. 커스텀 손실 함수

```python
# L1 손실 (MSE보다 더 선명한 결과)
train_loss, valid_loss, predictions = halfunet_decoding(
    embedding_train=embedding_train,
    embedding_valid=embedding_valid,
    label_train=label_train,
    label_valid=label_valid,
    loss_fn='l1',  # L1 loss
    num_epochs=100,
)

# BCE 손실 (이진 이미지의 경우)
train_loss, valid_loss, predictions = halfunet_decoding(
    embedding_train=embedding_train,
    embedding_valid=embedding_valid,
    label_train=binary_images_train,
    label_valid=binary_images_valid,
    loss_fn='bce',  # Binary cross-entropy
    num_epochs=100,
)
```

### 4. 멀티세션 데이터

```python
# 멀티세션 임베딩 (dict 형태)
embedding_train_multi = {
    0: [torch.randn(500, 512) for _ in range(5)],
    1: [torch.randn(500, 512) for _ in range(5)],
}

embedding_valid_multi = {
    0: [torch.randn(100, 512) for _ in range(5)],
    1: [torch.randn(100, 512) for _ in range(5)],
}

# 각 run에 대해 학습
train_loss, valid_loss, predictions = halfunet_decoding(
    embedding_train=embedding_train_multi,
    embedding_valid=embedding_valid_multi,
    label_train=label_train_multi,
    label_valid=label_valid_multi,
    n_run=0,  # 첫 번째 run
    num_epochs=100,
)
```

## 파라미터 가이드

### HalfUNetDecoder

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `input_dim` | int | 필수 | 입력 임베딩의 차원 |
| `output_channels` | int | 1 | 출력 채널 수 (1=그레이스케일, 3=RGB) |
| `output_shape` | tuple | (128, 256) | 출력 이미지 크기 (높이, 너비) |
| `latent_dim` | int | None | 내부 latent 차원. None이면 input_dim 사용 |

### halfunet_decoding

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `embedding_train` | Tensor/dict | 필수 | 학습 임베딩 |
| `embedding_valid` | Tensor/dict | 필수 | 검증 임베딩 |
| `label_train` | Tensor/dict | 필수 | 학습 타겟 이미지 |
| `label_valid` | Tensor/dict | 필수 | 검증 타겟 이미지 |
| `num_epochs` | int | 100 | 학습 에폭 수 |
| `lr` | float | 0.001 | 학습률 |
| `batch_size` | int | 32 | 배치 크기 |
| `device` | str | 'cuda' | 디바이스 ('cuda' 또는 'cpu') |
| `output_channels` | int | 1 | 출력 채널 수 |
| `output_shape` | tuple | (128, 256) | 출력 이미지 크기 |
| `latent_dim` | int | None | 내부 latent 차원 |
| `n_run` | int | None | 멀티세션의 경우 run 번호 |
| `loss_fn` | str | 'mse' | 손실 함수 ('mse', 'l1', 'bce') |

## 아키텍처 세부사항

### UNetDecoderOnly (latent_dim < 1024)
- 기본적인 UNet 디코더
- 5개의 업샘플링 블록
- Skip connections 사용
- 64x128 출력까지 지원

### ComplexLargeUNetDecoderOnly (latent_dim >= 1024)
- 고급 RRDB(Residual-in-Residual Dense Block) 사용
- PixelShuffle 기반 업샘플링 (더 선명한 결과)
- Channel Attention 메커니즘
- Edge/High-frequency head 추가
- 128x256 출력까지 지원

## 성능 최적화 팁

1. **Latent Dimension 선택**
   - 단순한 이미지: latent_dim = 256-512
   - 복잡한 이미지: latent_dim = 1024-2048
   - latent_dim >= 1024일 때 ComplexLargeUNetDecoderOnly 사용

2. **손실 함수 선택**
   - MSE: 전반적으로 부드러운 결과
   - L1: 더 선명한 엣지
   - BCE: 이진 이미지

3. **배치 크기와 학습률**
   - GPU 메모리에 따라 batch_size 조정
   - 큰 batch_size → 높은 학습률 (0.001-0.003)
   - 작은 batch_size → 낮은 학습률 (0.0001-0.001)

4. **학습 에폭**
   - 단순한 패턴: 50-100 epochs
   - 복잡한 이미지: 100-200 epochs
   - 조기 종료(early stopping) 권장

## 문제 해결

### ImportError 발생
```
ImportError: HalfUNet is not available
```
**해결**: kirby가 설치되어 있고 Python 경로에 있는지 확인
```python
import sys
sys.path.append('/path/to/CEBRA')
```

### 형상 불일치 오류
```
ValueError: Expected output with 3 channels, but got shape (batch, H, W)
```
**해결**: output_channels 파라미터와 실제 출력 채널이 일치하는지 확인

### 메모리 부족
**해결**:
- batch_size 줄이기
- latent_dim 줄이기
- gradient_checkpointing 사용 (PyTorch 1.11+)

### 저품질 재구성
**해결**:
- latent_dim 증가 (특히 1024 이상)
- 더 많은 epochs로 학습
- L1 손실 함수 시도
- 학습 데이터 증강

## 예제 코드 모음

전체 작동 예제는 `examples/halfunet_decoder_example.py`를 참조하세요.
