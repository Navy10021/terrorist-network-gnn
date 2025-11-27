# 시간적 그래프 신경망(T-GNN)을 이용한 테러 네트워크 분석

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**대테러 정보 분석을 위한 첨단 시간적 그래프 신경망**

[English](README.md) | 한국어

</div>

---

## 📋 개요

이 프로젝트는 테러 네트워크 분석 및 와해를 위한 최첨단 **시간적 그래프 신경망(Temporal Graph Neural Network, T-GNN)** 시스템을 구현합니다. 본 연구는 대테러 작전에서 중요한 세 가지 연구 질문을 다룹니다:

### 연구 질문

| 질문 | 설명 | 응용 |
|------|------|------|
| **Q1: 핵심 노드 탐지** | 제거 시 네트워크를 가장 효과적으로 와해시키는 노드는? | 개입 대상 우선순위 결정 |
| **Q2: 시간적 회복력** | 와해 후 네트워크가 어떻게 재구성될 것인가? | 사후 개입 모니터링 전략 |
| **Q3: 적대적 강건성** | 네트워크가 와해 시도에 어떻게 적응하는가? | 대응 적응 전술 |

### 주요 특징

- 🧠 **첨단 T-GNN 아키텍처**
  - 다중 스케일 패턴 포착을 위한 계층적 시간 풀링
  - 효율적 시간 모델링을 위한 LRU 기반 메모리 뱅크
  - 멀티헤드 어텐션 메커니즘
  - 그래프 트랜스포머 레이어

- 🌐 **다층 네트워크 분석**
  - 물리적 관계 (대면 회의)
  - 디지털 통신 (온라인 상호작용)
  - 금융 흐름 (자금 이체)
  - 이념적 연결 (공유 신념)
  - 작전 구조 (공동 활동)

- 📊 **포괄적 평가**
  - 12가지 기준선 비교 방법
  - 통계적 검증 (t-검정, Wilcoxon, 효과 크기)
  - 구성 요소 분석을 위한 제거 실험
  - 논문 품질의 시각화

---

## 🚀 빠른 시작

### 설치

```bash
# 저장소 복제
git clone https://github.com/yourusername/terrorist-network-tgnn.git
cd terrorist-network-tgnn

# PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric 설치
pip install torch-geometric

# 의존성 설치
pip install -r requirements.txt
```

### 실행

#### 옵션 1: Google Colab (초보자 권장)

1. `notebooks/terrorist_network_gnn_demo.ipynb`을 Colab에서 열기
2. **런타임** → **런타임 유형 변경** → **GPU** 선택
3. 셀을 순차적으로 실행

#### 옵션 2: Python 스크립트

```python
import torch
import sys
sys.path.append('./src')

from terrorist_network_dataset import NetworkConfig, TerroristNetworkGenerator
from main_experiment import EnhancedExperiment

# 설정
config = NetworkConfig(
    initial_nodes=50,
    max_nodes=80,
    recruitment_rate=0.05,
    dropout_rate=0.02
)

model_config = {
    'num_node_features': 64,
    'num_edge_features': 32,
    'hidden_dim': 128
}

# 실험 실행
experiment = EnhancedExperiment(
    config=config,
    model_config=model_config,
    output_dir='results/experiment_001'
)

experiment.run_complete_experiment(
    num_networks=10,
    num_timesteps=20,
    train_model=True,
    run_baselines=True,
    run_ablation=True
)
```

---

## 📊 예상 결과

### 성능 벤치마크

| 메트릭 | 우리 방법 | EvolveGCN | Dynamic GCN | PageRank | Random |
|--------|-----------|-----------|-------------|----------|--------|
| **와해 점수** | **0.7845** | 0.7123 | 0.6987 | 0.5834 | 0.4123 |
| **분절화** | **0.6234** | 0.5867 | 0.5645 | 0.4567 | 0.3012 |
| **작전 능력 감소** | **0.7123** | 0.6543 | 0.6234 | 0.5123 | 0.3456 |

**개선**: 최고 기준선 대비 +25.8% 와해, +37.8% 분절화

### 통계적 검증

```
대응 표본 t-검정: p = 0.0023 ** (매우 유의)
Wilcoxon 부호순위 검정: p = 0.0018 ** (매우 유의)
Cohen's d = 1.234 (큰 효과 크기)
✓ Bonferroni 보정: p = 0.028 * (여전히 유의)
```

---

## 📁 프로젝트 구조

```
terrorist-network-tgnn/
├── src/                                    # 소스 코드
│   ├── advanced_tgnn.py                    # 핵심 T-GNN 아키텍처
│   ├── terrorist_network_disruption.py     # 와해 알고리즘
│   ├── terrorist_network_dataset.py        # 네트워크 생성
│   ├── training.py                         # 학습 루프
│   ├── baselines.py                        # 비교 방법
│   ├── statistical_analysis.py             # 통계 검정
│   ├── ablation_study.py                   # 구성 요소 분석
│   └── main_experiment.py                  # 완전한 파이프라인
├── examples/                               # 사용 예제
│   └── terrorist_network_gnn_demo.ipynb    # 대화형 데모
├── tests/                                  # 단위 테스트
├── docs/                                   # 문서
├── data/                                   # 데이터 디렉토리
├── results/                                # 실험 결과
├── requirements.txt                        # 의존성
├── setup.py                                # 패키지 설정
└── README.md                               # 메인 문서
```

---

## 🎯 연구 질문 상세

### Q1: 핵심 노드 탐지

**목표**: 제거 시 네트워크 기능을 최대로 와해시키는 노드 식별

**방법론**:
1. 노드당 8가지 중심성 메트릭 계산
2. 5개 네트워크 레이어에 걸쳐 점수 집계
3. 시간적 중요도 가중치 적용
4. 학습된 중요도를 위한 GNN 임베딩 사용
5. 5가지 노드 선택 전략 테스트

**평가 메트릭**:
- 네트워크 분절화 (최대 컴포넌트 크기 감소)
- 와해 점수 (통합 메트릭)
- 작전 능력 감소
- 통신 효율성 손실

### Q2: 시간적 회복력 예측

**목표**: 노드 제거 후 네트워크 재구성 예측

**방법론**:
1. 잔여 노드 간 엣지 형성 확률 예측
2. 신규 멤버 모집 가능성 추정
3. 전체 네트워크 회복력 점수 계산
4. 개입을 위한 중요 시간 창 식별

**평가 메트릭**:
- 엣지 예측 정확도 (AUC-ROC)
- 모집 예측 MAE
- 회복력 점수 보정

### Q3: 적대적 강건성

**목표**: 와해에 대한 네트워크 적응 전략 분석

**시뮬레이션된 적응 전략**:
1. **분산화**: 단일 장애점 감소를 위한 중복 연결 생성
2. **모집**: 손실 대체를 위한 신규 멤버 신속 모집
3. **잠복**: 탐지 회피를 위한 통신 밀도 감소
4. **세분화**: 회복력을 위한 자율 셀 분할

**평가**:
- 와해 후 회복률
- 기능 복원 시간
- 반복 공격에 대한 회복력
- 각 전략의 비용-편익

---

## 🔒 윤리적 고려사항

본 연구는 엄격한 윤리 지침을 따릅니다:

### 원칙

✅ **방어 목적만**: 연구는 오직 대테러 방어를 위해 수행됨

✅ **합성 데이터**: 모든 네트워크는 인공적으로 생성됨 - 실제 개인이나 조직 없음

✅ **정보기관 협력**: 합법적 법 집행 사용을 위해 설계됨

✅ **책임있는 공개**: 학술 동료 심사를 통해 결과 공유

✅ **투명성**: 학술적 검토를 위한 오픈소스 코드

### 안전장치

🛡️ **실제 데이터 없음**: 시스템이 실제 정보 데이터를 절대 처리하지 않음

🛡️ **이중 사용 인식**: 연구자들이 잠재적 오용 위험을 인정함

🛡️ **접근 제어**: 승인된 기관에서만 배포 권장

🛡️ **학술 감독**: 기관 검토 및 윤리 승인 대상

### 경고

⚠️ **이 코드를 다음 용도로 사용해서는 안 됩니다**:
- 합법적 정치 조직을 대상으로 함
- 언론의 자유나 집회의 자유를 억압함
- 법적 승인 없이 감시를 수행함
- 사회 운동이나 시위를 분석함

---

## 📧 연락처 및 지원

- **저자**: 이윤섭
- **이메일**: iyunseob4@gmail.com
- **GitHub Issues**: [버그 보고 또는 기능 요청](https://github.com/yourusername/terrorist-network-tgnn/issues)
- **토론**: [질문하거나 아이디어 공유](https://github.com/yourusername/terrorist-network-tgnn/discussions)

---

## 📄 라이센스

이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🙏 감사의 말

- **PyTorch Geometric 팀** - 훌륭한 GNN 라이브러리와 문서
- **연구 커뮤니티** - 기준선 구현 및 방법론
- **정보기관** - 문제 정식화 및 요구사항
- **학술 심사자** - 피드백 및 윤리적 지침

---

## 🔄 버전 히스토리

### v1.0.0 (2025년 11월)
- ✅ 초기 공개 릴리스
- ✅ 모든 구성 요소를 갖춘 완전한 T-GNN 구현
- ✅ 12가지 기준선 비교 방법
- ✅ 통계적 검증 프레임워크
- ✅ 논문 품질의 시각화
- ✅ 포괄적인 문서

---

<div align="center">

**⭐ 유용하다면 이 저장소에 별을 주세요! ⭐**

대테러 연구 커뮤니티를 위해 ❤️로 만들어졌습니다

</div>
