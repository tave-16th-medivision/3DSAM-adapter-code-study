# 3DSAM-adapter Code Study 

본 리포지토리는 "3DSAM-Adapter: A 3D Adapter for SAM in Medical Image Segmentation" 논문의 코드를 분석하고, 2D 기반의 SAM(Segment Anything Model)을 3D 볼륨 데이터에 효율적으로 적용하는 방법을 학습하기 위한 스터디입니다.

This repository is a personal code study for analyzing the implementation of the "3DSAM-Adapter" paper. The goal is to understand how to efficiently adapt the pre-trained 2D SAM for 3D volumetric data (e.g., medical images) using parameter-efficient fine-tuning (PEFT).

## 1. 원본 자료 (Original Resources)

* **Paper:** [3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation (arXiv:2306.13465)](https://arxiv.org/abs/2306.13465)
* **Original GitHub:** [med-air/3DSAM-adapter](https://github.com/med-air/3DSAM-adapter)


## 2. 분석 범위 (Scope of Study)

본 스터디는 3DSAM-adapter의 핵심 아이디어인 '어댑터 모듈'의 구현과 3D 데이터 처리, 그리고 학습/추론 파이프라인 전반을 분석합니다.

This study analyzes the entire pipeline, including the 3D data handling, the implementation of the core 'Adapter' module, and the training/inference scripts.

* `modeling/`: SAM 원본 모델 및 3D 어댑터 모듈 정의
* `dataset/`: 3D 볼륨 데이터셋 로더 및 전처리
* `train.py` / `train_auto.py`: 모델 학습 스크립트
* `test.py` / `test_auto.py`: 모델 평가 스크립트
* `utils/`: 각종 유틸리티 (메트릭, 설정 등)


## 3. 핵심 아키텍처 분석 (Core Architecture Analysis)

### 문제점 (The Problem)

SAM은 2D 이미지(e.g., `[3, 1024, 1024]`)용으로 사전 훈련되었습니다. 3D 볼륨 데이터(e.g., MRI, `[D, H, W]`)에 이를 직접 적용하려면 문제가 발생합니다.

* **Naive 2D:** 3D 볼륨을 2D 슬라이스(slice)로 쪼개어 개별 처리하면, **슬라이스 간의 3차원 공간적 문맥(inter-slice spatial context)**을 잃게 됩니다.
* **Full 3D Fine-tuning:** SAM의 모든 가중치(수억 개)를 3D 데이터로 재학습시키는 것은 엄청난 연산 비용이 들며, SAM이 가진 원래의 일반화(generalization) 성능을 잃을 수 있습니다(catastrophic forgetting).

### 해결책: 3DSAM-adapter (The Solution)

이 모델은 SAM의 거대한 사전 훈련 가중치(Image Encoder, Prompt Encoder, Mask Decoder)를 **전부 동결(freeze)**하고, 각 컴포넌트(특히 Image Encoder의 ViT)에 **작고 가벼운 '어댑터(Adapter)' 모듈**을 삽입합니다.

This model **freezes** the entire pre-trained SAM (Image Encoder, Prompt Encoder, and Mask Decoder) and injects small, lightweight **Adapter modules** into the components, especially within the Image Encoder's Vision Transformer (ViT).

* **학습 대상 (Trainable Parameters):** 오직 새로 삽입된 **어댑터 모듈**만 학습 대상이 됩니다.
* **Holistic Adaptation (총체적 적응):** 논문 제목처럼, 어댑터는 Image Encoder뿐만 아니라 Prompt Encoder와 Mask Decoder에도 선택적으로 추가되어 3D 태스크에 맞게 모델 전체를 '총체적으로' 미세 조정합니다.
* **기능 (Function):** 이 어댑터 모듈이 3D 데이터의 **슬라이스 간(inter-slice) 정보**를 학습하여 2D 피처에 3D 문맥을 주입합니다.
* **결과 (Result):** 전체 파라미터의 극히 일부(e.g., < 1%)만으로 3D 태스크에 대한 높은 성능을 달성하는 **Parameter-Efficient Fine-Tuning (PEFT)** 를 실현합니다.


## 4. 주요 파일 및 디렉토리 분석 (File & Directory Analysis)

### `modeling/`

* 

### `dataset/`

* 

### `train.py` / `train_auto.py`

* 

### `test.py` / `test_auto.py`

* 

### `utils/`

* 
