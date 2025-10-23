# Allen Brain Observatory Dataset - CEBRA Integration

이 문서는 Allen Brain Observatory calcium imaging 데이터를 CEBRA의 poyo-ssl 스타일로 로드하고 학습하는 방법을 설명합니다.

## 📁 파일 구조

```
CEBRA/
├── data/                           # Allen Brain Observatory .h5 파일들
│   ├── 501021421.h5
│   ├── 501574836.h5
│   └── ...
├── allen_config.yaml               # 설정 파일
├── run_allen_cebra.py              # 메인 실행 스크립트 (batch_size=4)
├── simple_allen_test.py            # 간단한 데이터 로딩 테스트
├── test_allen_dataset.py           # 전체 데이터셋 구조 탐색
├── test_cebra_allen.py             # CEBRA 통합 테스트
└── results/                        # 결과 저장 디렉토리
```

## 📊 데이터셋 정보

### Allen Brain Observatory Visual Coding - Calcium Imaging

- **Neural Data**: df_over_f (ΔF/F) calcium traces
- **Shape**: (timepoints, neurons)
  - Session 1: 32,187 timepoints × 150 neurons
  - Session 2: 32,243 timepoints × 240 neurons
  - Session 3: 32,215 timepoints × 227 neurons
  - Session 4: 32,235 timepoints × 181 neurons
- **Splits**: train_mask, valid_mask, test_mask 포함
- **Temporal Domain**: 시작/끝 timestamp 포함

### 데이터 특징

- ✅ Train/Valid/Test split이 이미 분리되어 있음
- ✅ 각 세션마다 다른 수의 뉴런 (150~240개)
- ✅ Natural movie one stimulus 데이터 포함
- ✅ Drifting gratings stimulus 데이터 포함

## 🚀 빠른 시작

### 1. 간단한 데이터 로딩 테스트

```bash
python3 simple_allen_test.py
```

이 스크립트는:
- ✅ .h5 파일 로드 확인
- ✅ Train/Valid/Test split 확인
- ✅ PyTorch DataLoader 생성 (batch_size=4)
- ✅ 여러 세션 동시 로드 테스트

### 2. 메인 학습 스크립트 실행

```bash
# 기본 설정으로 실행
python3 run_allen_cebra.py --config allen_config.yaml

# 4개 세션으로 제한
python3 run_allen_cebra.py --config allen_config.yaml --max_sessions 4

# Batch size 변경
python3 run_allen_cebra.py --config allen_config.yaml --batch_size 8

# 데이터 디렉토리 변경
python3 run_allen_cebra.py --config allen_config.yaml --data_dir /path/to/data
```

## ⚙️ 설정 파일 (allen_config.yaml)

### 주요 설정

```yaml
# 데이터셋 설정
dataset:
  data_dir: "data"
  file_pattern: "*.h5"

# CEBRA 모델 설정 (poyo-ssl style)
model:
  batch_size: 4                    # Batch size
  learning_rate: 0.0003            # Learning rate
  output_dimension: 32             # 임베딩 차원
  max_iterations: 5000             # 최대 반복 횟수
  learning_mode: "time_contrastive" # Self-supervised learning
  conditional: "time_delta"        # 시간 정보 활용

# 학습 설정
training:
  max_epochs: 100
  early_stopping:
    enabled: true
    patience: 10
```

## 📝 스크립트 설명

### `simple_allen_test.py`
- 의존성이 최소화된 간단한 테스트
- PyTorch 기본 DataLoader만 사용
- 데이터 로딩 확인용

**실행 결과:**
```
✅ All tests passed!
   Total sessions loaded: 4
   Total neurons: 798
   Total timepoints: 128880
   Average neurons per session: 199.5
```

### `run_allen_cebra.py`
- 메인 학습 스크립트
- Multi-session 처리
- Config 파일 기반 설정
- 결과 자동 저장

**주요 기능:**
- ✅ 여러 세션 동시 처리 (separate mode)
- ✅ Batch size=4로 데이터 로딩
- ✅ Train/Valid split 자동 처리
- ✅ Z-score normalization
- ✅ Timestamp normalization

## 🔧 데이터 로딩 모드

스크립트는 세 가지 모드를 지원합니다:

### 1. `separate` (기본값) - 권장
```python
# 각 세션을 독립적으로 처리
# Multi-session CEBRA에 적합
Session 1: (32187, 150)
Session 2: (32243, 240)
Session 3: (32215, 227)
Session 4: (32235, 181)
```

### 2. `pad`
```python
# 모든 세션을 최대 뉴런 수에 맞춰 padding
# Single-session처럼 처리 가능
All sessions: (128880, 240)  # 240 = max neurons
```

### 3. `first_only`
```python
# 첫 번째 세션만 사용
# 빠른 테스트용
Session 1: (32187, 150)
```

## 📈 예상 출력

### Batch 구조 (batch_size=4)

```
Session 1:
   Batch 1: neural=torch.Size([4, 150]), time=torch.Size([4])
   Batch 2: neural=torch.Size([4, 150]), time=torch.Size([4])
   Batch 3: neural=torch.Size([4, 150]), time=torch.Size([4])

Session 2:
   Batch 1: neural=torch.Size([4, 240]), time=torch.Size([4])
   ...
```

## 🔍 데이터 검증

### 데이터 구조 확인
```bash
python3 test_allen_dataset.py
```

출력 예시:
```
Calcium traces analysis:
   - df_over_f shape: (115459, 150)
   - train_mask: 32187 / 115459 samples
   - valid_mask: 4486 / 115459 samples
   - test_mask: 8064 / 115459 samples
   - domain: [5.60919, 3845.695]
```

## 🐛 문제 해결

### kirby 의존성 에러가 있는 경우
```bash
# NestedEnumType __prepare__ 에러는 이미 수정됨
# kirby/taxonomy/core.py에 __prepare__ 메서드 추가됨
```

### CEBRA import 에러가 있는 경우
```bash
# simple_allen_test.py 사용 (CEBRA 없이 작동)
python3 simple_allen_test.py
```

### 세션마다 뉴런 수가 다른 경우
```bash
# separate mode 사용 (기본값)
python3 run_allen_cebra.py --config allen_config.yaml

# 또는 pad mode로 통일
# run_allen_cebra.py의 train_model 함수에서 mode='pad' 지정
```

## 📚 다음 단계

### 서버에서 실행하기

1. **파일 전송**
```bash
# 로컬에서 서버로
scp -r data/ allen_config.yaml run_allen_cebra.py server:/path/to/CEBRA/
```

2. **서버에서 실행**
```bash
ssh server
cd /path/to/CEBRA
python3 run_allen_cebra.py --config allen_config.yaml --max_sessions 4
```

3. **GPU 사용 (서버에서)**
```yaml
# allen_config.yaml에서
model:
  device: "cuda"  # 또는 "cuda_if_available"
```

### CEBRA 전체 통합

```python
# Full CEBRA integration 예시
import cebra
from cebra.data import TensorDataset, DatasetCollection

# 데이터 로드 (run_allen_cebra.py 참고)
train_datasets = load_dataset(config, split='train')

# CEBRA 모델 생성
cebra_model = cebra.CEBRA(
    model_architecture='offset10-model',
    batch_size=4,
    learning_rate=3e-4,
    temperature=1,
    output_dimension=32,
    max_iterations=5000,
    distance='cosine',
    conditional='time_delta',
    device='cuda_if_available',
    verbose=True,
)

# 학습
cebra_model.fit(dataset_collection)

# 임베딩 추출
embeddings = cebra_model.transform(dataset)
```

## 📊 성능 정보

### 데이터 크기
- **총 세션**: 16개 (전체)
- **테스트용**: 4개 세션
- **총 뉴런**: ~798개 (4개 세션)
- **총 timepoints**: ~128,880개 (train)
- **배치 수/에폭**: ~8,000 batches (session당, batch_size=4)

### 메모리 사용
- Session당 약 100-300MB
- 4개 세션 동시: ~1GB
- GPU 메모리: 2-4GB 권장

## ✅ 검증 완료 항목

- [x] .h5 파일 구조 탐색
- [x] Allen Brain Observatory 데이터 형식 확인
- [x] Train/Valid/Test split 확인
- [x] Multi-session 데이터 로딩
- [x] Batch size=4로 DataLoader 생성
- [x] Config 파일 작성
- [x] 실행 스크립트 작성 및 테스트
- [x] kirby taxonomy 에러 수정
- [x] 서버 이전 준비 완료

## 🎯 요약

모든 파일이 준비되었습니다:

1. **설정 파일**: `allen_config.yaml`
2. **실행 스크립트**: `run_allen_cebra.py` (batch_size=4)
3. **테스트 스크립트**: `simple_allen_test.py`

서버로 옮겨서 바로 실행할 수 있습니다! 🚀
