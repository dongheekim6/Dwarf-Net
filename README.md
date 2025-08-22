# Dwarf-Net

Dwarf-Net은 **소형 용종 분할(Small Polyp Segmentation)** 성능을 개선하기 위해 설계된 경량형 의료 영상 세그멘테이션 모델입니다.  
기존 U-Net 기반 구조의 한계를 극복하고, 작은 크기의 병변에서도 높은 정확도를 보장하는 것을 목표로 합니다.

---

## 🚀 주요 특징
- **Shallow Decoder**: 불필요한 연산을 줄이고 작은 객체에 집중
- **고해상도 Skip Connection**: 세밀한 경계 보존
- **다양한 Context Module 지원**: Local/Global 컨텍스트 통합

---

## 📂 프로젝트 구조
```bash
modelcombination/
├── models/                # 모델 정의
├── splits/                # 데이터셋 분할 스크립트
├── 250707_train.py        # 학습 스크립트
├── evaluate.py            # 평가 스크립트
├── utils.py               # 유틸리티 함수
├── requirements.txt       # 필요 패키지
└── README.md
