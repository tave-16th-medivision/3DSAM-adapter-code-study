"""
의료 영상(CT, MRI 등) 3D 볼륨 데이터셋 정의 및 DataLoader 구성 파일
"""

import pickle  # 데이터 split 정보(pkl 파일) 로드용
import os, sys
from torch.utils.data import DataLoader, Dataset  # PyTorch 데이터 로딩 유틸리티
import torch
import numpy as np
import nibabel as nib  # 의료 영상 파일(NIfTI 등) 입출력
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset  # 공통 3D 볼륨 처리 로직을 담은 부모 클래스

# ------------------------------------------------------------
# 개별 데이터셋 클래스 정의
# (각기 다른 의료 영상 데이터셋별 intensity 범위, 표준화 값 등 정의)
# ------------------------------------------------------------

class KiTSVolumeDataset(BaseVolumeDataset):
    """KiTS (Kidney Tumor Segmentation) 데이터셋 정의"""
    def _set_dataset_stat(self):
        # 원본 shape 순서: (depth, height, width)
        # org load shape: d, h, w
        self.intensity_range = (-54, 247)  # intensity 범위 (HU 값 등)
        self.target_spacing = (1, 1, 1)    # 리샘플링 간격 (mm 단위)
        self.global_mean = 59.53867        # intensity 정규화용 평균값
        self.global_std = 55.457336        # intensity 정규화용 표준편차
        self.spatial_index = [0, 1, 2]     # (z, y, x) → (D, H, W) 변환 인덱스
        self.do_dummy_2D = False           # 2D dummy crop 사용 여부
        self.target_class = 2              # 세그멘테이션할 타겟 클래스 인덱스 (신장 종양) 


class LiTSVolumeDataset(BaseVolumeDataset):
    """LiTS (Liver Tumor Segmentation) 데이터셋 정의"""
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    """Pancreas (췌장) 세그멘테이션 데이터셋 정의"""
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-39, 204)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True   # 췌장 데이터셋은 2D crop 병행 사용
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset):
    """Colon (대장) 세그멘테이션 데이터셋 정의"""
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-57, 175)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True   # 대장 데이터셋은 2D crop 병행 사용
        self.target_class = 1

# ------------------------------------------------------------
# 데이터셋 이름 → 클래스 매핑 딕셔너리
# ------------------------------------------------------------
DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "pancreas": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
}

# ------------------------------------------------------------
# DataLoader 구성 함수
#   - pickle에 저장된 train/val/test split 정보 로드
#   - BaseVolumeDataset 기반의 Dataset 인스턴스 생성
#   - torch.utils.data.DataLoader로 래핑 후 반환
# ------------------------------------------------------------
def load_data_volume(
    *,
    data,                       # 데이터셋 이름 ("kits", "lits", "pancreas", "colon")
    path_prefix,                # 데이터셋 루트 디렉토리 경로
    batch_size,                 # 배치 크기
    data_dir=None,              # split.pkl 파일 경로 (기본값: path_prefix/split.pkl)
    split="train",              # 데이터 분할 ("train", "val", "test")
    deterministic=False,        # deterministic 모드 여부 (shuffle 여부)
    augmentation=False,         # 데이터 증강 사용 여부
    fold=0,                     # 교차 검증용 fold index
    rand_crop_spatial_size=(96, 96, 96),  # 랜덤 crop 크기
    convert_to_sam=False,       # SAM 모델용 데이터 변환 여부
    do_test_crop=True,          # 테스트 시 crop 수행 여부
    do_val_crop=True,           # 검증 시 crop 수행 여부
    do_nnunet_intensity_aug=False, # nnUNet 스타일 intensity augmentation 여부
    num_worker=4,               # DataLoader 병렬 로딩 스레드 개수
):
    # 데이터 루트 경로 미지정 시 오류 처리
    if not path_prefix:
        raise ValueError("unspecified data directory")
    
    # split.pkl 파일 경로 자동 지정
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")

    # split.pkl 로드 (pickle 구조: [fold][split] 형태의 dict)
    with open(data_dir, "rb") as f: 
        d = pickle.load(f)[fold][split]

    # 이미지와 세그멘테이션 파일 경로 리스트 구성
    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    # 선택한 데이터셋 클래스 생성
    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    # DataLoader 구성 (deterministic 여부에 따라 shuffle 제어)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )

    # 최종 DataLoader 반환
    return loader