# MLOPS Study - Modularization


## Toy example - Debt Default Prediction (Dacon Educational Hackathon)

본 프로젝트는 Dacon 해커톤 ** “채무 불이행 여부 예측 AI 알고리즘 개발” **을 목표로, 딥러닝 모델 기반 예측 파이프라인을 구축했습니다.  
**MLOps 초기 수준의 모듈화 구조**를 도입이 목표이며, 관리 용이성 및 확장성 등을 고려했습니다.

---

## ✅ 1. 프로젝트 구조 및 모듈 설명

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

## 🔁 2. 데이터 및 워크플로우 흐름

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

## ⚖️ 3. 모듈화의 장단점

### 장점

- **유지보수 용이**: 기능별 분리로 코드 가독성 및 수정 용이
- **재사용성 확보**: 전처리, 모델, 학습 루프 등을 다양한 실험에 활용 가능
- **설정 관리 효율화**: `config.yaml`을 통해 하이퍼파라미터 및 경로 관리 일원화
- **확장성 우수**: 모델 구조 변경, 앙상블, AutoML 등 실험 확장에 적합

### 단점

- **초기 진입 장벽**: 디렉토리 구조와 모듈 연결에 대한 이해 필요
- **오버엔지니어링 가능성**: 단일 스크립트 수준의 간단한 실험엔 다소 복잡할 수 있음
- **디버깅 난이도 상승**: 오류 발생 시 여러 모듈 추적 필요

---

## TO-DO List

- `wandb`, `MLflow` 등 실험 추적 도구 도입
- 모델 버전 관리 및 저장 기능 추가
- 다른 ML 모델(LGBM, XGBoost)과의 성능 비교 및 앙상블
- 스터디원들로 부터 Feedback 받기

---
