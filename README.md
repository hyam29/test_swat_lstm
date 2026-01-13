# SWaT 시계열 이상탐지 (LSTM AutoEncoder)

**SWaT 시계열 센서 데이터**를 대상으로  
**LSTM AutoEncoder 기반 이상탐지 모델**을 구현하고 실험한 저장소

정상 데이터로 모델을 학습한 뒤,  
슬라이딩 윈도우 단위의 **재구성 오차(Reconstruction Error)**를 이용해  
이상 구간을 탐지한다.

---

## 프로젝트 목적
- 시계열 데이터 기반 이상탐지 파이프라인 구현
- 학습/평가 코드 구조 분리 및 실험 재현성 확보
- LSTM AutoEncoder의 이상탐지 성능 분석

---

## 데이터 개요
- 다변량 시계열 센서 데이터 (SWaT)
- 라벨:
  - `0`: 정상(normal)
  - `1`: 어택(attach)
- 슬라이딩 윈도우 방식으로 입력 구성  
※ 원본 데이터는 포함하지 않음

---

## 모델 및 방법
- 모델: **LSTM AutoEncoder**
- 입력: 시계열 윈도우 `(T, F)`
- 학습: 정상 데이터만 사용
- 평가:
  - 윈도우 단위 재구성 오차 계산
  - 정상 데이터 기반 threshold 설정
  - Precision / Recall / F1, ROC-AUC, PR-AUC

---

## 프로젝트 구조
├─ src/ # 모델, 데이터셋, 학습 코드
├─ notebooks/ # 실험 및 평가 노트북
├─ configs/ # 학습 설정 파일
├─ runs/ # 학습 결과 (git 제외)
└─ data/ # 원본/전처리 데이터 (git 제외)
