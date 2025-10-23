# Kirby Dataset Integration Guide

이 가이드는 CEBRA에서 kirby의 Dataset을 사용하는 방법을 설명합니다.

## 개요

CEBRA에 kirby Dataset을 통합하여 kirby의 강력한 데이터 관리 기능을 CEBRA의 학습 파이프라인과 함께 사용할 수 있습니다. 이를 위해 두 가지 새로운 클래스가 추가되었습니다:

1. **KirbyDatasetAdapter**: kirby Dataset의 개별 세션을 CEBRA의 SingleSessionDataset으로 변환
2. **DatasetCollection.from_kirby_dataset()**: kirby Dataset을 CEBRA의 DatasetCollection으로 쉽게 변환하는 클래스 메서드

## 기본 사용법

### 1. 단일 세션 사용하기

```python
from kirby.data import Dataset as KirbyDataset
from cebra.data.datasets import KirbyDatasetAdapter

# kirby Dataset 생성
kirby_ds = KirbyDataset(
    root="data",
    split="train",
    include=[{
        "selection": [{
            "dandiset": "allen_brain_observatory_calcium",
        }]
    }],
    task='movie_decoding_one',
)

# 특정 세션을 CEBRA 형식으로 변환
session_id = kirby_ds.session_ids[0]
cebra_dataset = KirbyDatasetAdapter(
    kirby_dataset=kirby_ds,
    session_id=session_id,
    continuous_keys=['timestamps'],  # 연속 변수로 사용할 키
    discrete_keys=['unit_cre_line'],  # 이산 변수로 사용할 키
    neural_key='patches',  # 신경 데이터 키
    device='cuda',  # 또는 'cpu'
)

# CEBRA의 데이터로더와 함께 사용
from cebra.data.single_session import ContinuousDataLoader

loader = ContinuousDataLoader(
    dataset=cebra_dataset,
    num_steps=10,
    batch_size=512,
)
```

### 2. 다중 세션 컬렉션 사용하기

```python
from kirby.data import Dataset as KirbyDataset
from cebra.data import DatasetCollection

# kirby Dataset 생성 (여러 세션 포함)
kirby_ds = KirbyDataset(
    root="data",
    split="train",
    include=[{
        "selection": [{
            "dandiset": "allen_brain_observatory_calcium",
        }]
    }],
    task='movie_decoding_one',
)

# 모든 세션을 CEBRA DatasetCollection으로 변환
collection = DatasetCollection.from_kirby_dataset(
    kirby_dataset=kirby_ds,
    continuous_keys=['timestamps'],
    discrete_keys=['unit_cre_line'],
    neural_key='patches',
    device='cuda',
)

# 또는 특정 세션만 선택
collection = DatasetCollection.from_kirby_dataset(
    kirby_dataset=kirby_ds,
    continuous_keys=['timestamps'],
    session_ids=['allen_brain_observatory_calcium/501940850',
                 'allen_brain_observatory_calcium/502608215'],
    device='cuda',
)

# 다중 세션 학습에 사용
from cebra.data.multi_session import ContinuousMultiSessionDataLoader

multi_loader = ContinuousMultiSessionDataLoader(
    dataset=collection,
    num_steps=10,
    batch_size=512,
)
```

## 파라미터 설명

### KirbyDatasetAdapter

- **kirby_dataset**: kirby.data.Dataset 인스턴스
- **session_id**: 추출할 세션의 ID (예: 'allen_brain_observatory_calcium/501940850')
- **continuous_keys**: 연속 변수로 사용할 kirby Data 객체의 키 리스트
  - 예: `['timestamps']`, `['timestamps', 'position']`
- **discrete_keys**: 이산 변수로 사용할 kirby Data 객체의 키 리스트
  - 예: `['unit_cre_line']`, `['session_index']`
- **neural_key**: 신경 데이터에 접근할 키 (기본값: 'patches')
- **device**: 데이터를 저장할 디바이스 (기본값: 'cpu')

### DatasetCollection.from_kirby_dataset()

위의 파라미터에 추가로:

- **session_ids**: 포함할 특정 세션 ID 리스트 (선택사항)
  - None인 경우 모든 세션이 포함됩니다

## 고급 사용 예제

### 1. 커스텀 변환과 함께 사용

```python
from cebra.data import DatasetCollection

# kirby Dataset에서 DatasetCollection 생성
collection = DatasetCollection.from_kirby_dataset(
    kirby_dataset=kirby_ds,
    continuous_keys=['timestamps', 'latent_timestamps'],
    discrete_keys=['unit_cre_line', 'session_index'],
    neural_key='patches',
)

# 개별 세션에 접근
session_0 = collection.get_session(0)
print(f"Session 0 input dimension: {session_0.input_dimension}")
print(f"Session 0 length: {len(session_0)}")
```

### 2. UnifiedDataset과 함께 사용

```python
from cebra.data.datasets import UnifiedDataset

# 여러 세션을 하나의 pseudo-session으로 통합
unified = UnifiedDataset.from_kirby_dataset(
    kirby_dataset=kirby_ds,
    continuous_keys=['timestamps'],
    device='cuda',
)
```

## 주의사항

1. **데이터 키 확인**: kirby Dataset의 Data 객체가 지정한 키를 가지고 있는지 확인하세요.
   ```python
   session_data = kirby_ds.get_session_data(session_id)
   print("Available keys:", session_data.keys)  # 사용 가능한 키 확인
   ```

2. **메모리 관리**: `get_session_data()`는 전체 세션을 메모리에 로드할 수 있으므로 주의하세요.

3. **디바이스 일관성**: 모든 세션이 동일한 디바이스에 있어야 합니다.

4. **인덱스 타입**:
   - continuous_keys는 float 타입 데이터를 가리켜야 합니다
   - discrete_keys는 integer 타입 데이터를 가리켜야 합니다

## 문제 해결

### ImportError 발생 시
```
ImportError: kirby package is not available
```
kirby가 설치되어 있고 Python 경로에 있는지 확인하세요:
```python
import sys
sys.path.append('/path/to/CEBRA')  # kirby 폴더가 있는 경로
```

### 키가 없다는 오류 발생 시
```
ValueError: Session data does not contain neural key 'patches'
```
사용 가능한 키를 확인하고 올바른 키를 지정하세요:
```python
session_data = kirby_ds.get_session_data(session_id)
print(session_data.keys)
```

## 예제 코드 모음

전체 작동 예제는 `examples/kirby_integration_example.py`를 참조하세요.
