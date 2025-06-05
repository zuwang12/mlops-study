# MLOPS Study - Modularization


## 1. 개요(Toy example) - Debt Default Prediction (Dacon)

본 프로젝트는 Dacon 해커톤 "**채무 불이행 여부 예측 AI 알고리즘 개발**"을 Target으로, 딥러닝 기반 예측 파이프라인을 구축했습니다.  
**MLOps 모듈화 구조**를 구성이 목표이며, 관리 용이성 및 확장성 등을 고려했습니다.

---

## 2. 프로젝트 구조 및 모듈 설명

본 프로젝트는 기능별로 디렉토리와 모듈을 분리하여 다음과 같은 구조로 구성되어 있습니다:

```
mlops/
├── configs/              # 설정 파일 (YAML)
│   └── config.yaml
├── data/                 # 데이터 CSV 파일 (train.csv, test.csv)
├── datasets/             # PyTorch Dataset 정의
│   └── tabular_dataset.py
├── models/               # 모델 구조 정의
│   └── JK_model.py
├── utils/                # 전처리 및 보조 함수
│   ├── preprocessing.py
│   └── metrics.py
├── train.py              # 학습 루프
├── predict.py            # 추론 및 제출 저장
├── main.py               # 전체 파이프라인 실행 진입점
└── submission_JK.csv     # 최종 제출 파일
```

| 모듈 | 설명 |
|------|------|
| `main.py` | 전체 파이프라인 실행 진입점 (train → inference) |
| `configs/config.yaml` | 경로, 하이퍼파라미터 등을 외부 설정 파일로 관리 |
| `datasets/tabular_dataset.py` | PyTorch 기반 Dataset 클래스 정의 |
| `models/JK_model.py` | DNN 기반의 `JKModel` 구조 정의 (BatchNorm + Dropout 포함) |
| `utils/preprocessing.py` | Label Encoding, StandardScaler 기반 전처리 로직 |
| `utils/metrics.py` | ROC-AUC 평가 함수 (선택사항) |
| `train.py` | 학습 루프, Validation 평가, 모델 학습/저장 |
| `predict.py` | 학습된 모델을 통한 테스트셋 추론 및 결과 저장 |

---

## 3. 데이터 및 워크플로우 흐름

### 📊 전체 데이터 흐름

```
[data/train.csv, data/test.csv]
         ↓
  [전처리: preprocessing.py]
         ↓
[TabularDataset → DataLoader 변환]
         ↓
[모델 학습: train.py]
         ↓
[검증: Validation ROC-AUC 출력]
         ↓
[테스트 추론: predict.py]
         ↓
[결과 저장: submission_JK.csv]
```

### ⚙️ 전체 실행 흐름

```
1. main.py 실행
2. config.yaml 설정 로드
3. train.csv → 전처리 후 학습/검증 데이터 분할
4. JKModel 학습 (epoch 반복)
5. 검증셋 AUC 평가 출력
6. test.csv에 대해 추론 수행
7. UID와 예측 결과 저장 → submission_JK.csv 생성
```

---

## 4. 모듈화의 장단점


### 장점

#### 유지보수 용이
- 기능별로 파일을 분리함으로써 코드 가독성과 유지보수성이 높아졌습니다.
- 예를 들어, 모델 구조를 수정할 경우 `models/JK_model.py`만 수정하면 되고, 학습(`train.py`)이나 추론(`predict.py`) 코드는 그대로 유지할 수 있습니다.

```python
# models/JK_model.py
class JKModel(nn.Module):
    def __init__(self, input_dim):
        ...
```

#### 재사용성 확보
- `utils/preprocessing.py`에 정의된 전처리 함수는 학습과 추론 양쪽에서 재사용됩니다.
- 전처리 방식 변경 시에도 한 파일만 수정하면 되므로 코드 일관성을 유지할 수 있습니다.

```python
# utils/preprocessing.py
def preprocess(train_df, test_df, target_col='채무 불이행 여부'):
    ...
```

#### 설정 관리 효율화
- 실험에 필요한 하이퍼파라미터와 경로 등을 `configs/config.yaml`로 외부화하여 관리합니다.
- 여러 실험을 반복할 때 코드를 수정하지 않고 설정 파일만 바꾸면 됩니다.

```yaml
# configs/config.yaml
train_path: "./data/train.csv"
batch_size: 64
epochs: 10
lr: 0.001
```

```python
# main.py
cfg = load_config()
train_df = pd.read_csv(cfg['train_path'])
```

#### 확장성 우수
- 새로운 모델 추가, 앙상블 실험, AutoML 도입 등 구조 확장이 용이합니다.
- 모델 이름에 따라 실행될 클래스를 바꾸는 방식으로 `train.py`를 확장할 수 있습니다.

```python
# train.py (예시)
if cfg['model'] == 'JK':
    model = JKModel(...)
elif cfg['model'] == 'LGBM':
    model = LGBMClassifier(...)
```

---

### 단점

#### 초기 진입 장벽
- `main.py`부터 시작해 `train.py`, `models/`, `utils/`로 이어지는 호출 흐름을 처음 접한 사람은 전체 구조를 이해하는 데 시간이 필요합니다.

```
실행 흐름 예시:
main.py → train.py → models/JK_model.py → utils/preprocessing.py → datasets/tabular_dataset.py
```

#### 오버엔지니어링 가능성
- 간단한 실험에도 구조가 분리되어 있어 단일 스크립트보다 복잡해질 수 있습니다.
- 예: 단일 `.py`로 구현 가능한 실험이 5개 이상의 파일로 나뉘어 있음

```
main.py → train.py → model.py + preprocessing.py + dataset.py
```

#### 디버깅 난이도 상승
- 예측 결과 이상 발생 시 여러 모듈을 거슬러 올라가며 원인을 추적해야 합니다.
- 학습 결과가 이상하면 모델 구조, 데이터 전처리, Dataset 등 여러 파일을 확인해야 합니다.

```python
# train.py
output = model(xb)  # model, dataset, preprocessing 모두 연결됨
```

---

## 5. TO-DO List

- `wandb`, `MLflow` 등 실험 추적 도구 도입
- 모델 버전 관리 및 저장 기능 추가
- 다른 ML 모델(LGBM, XGBoost)과의 성능 비교 및 앙상블
- 스터디원들로 부터 Feedback 받기

---
