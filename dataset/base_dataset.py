"""
의료 영상(CT, MRI 등)의 3D 볼륨 데이터를 로드하고,
MONAI 변환(transform)을 적용하는 기본 Dataset 클래스 정의 파일
"""

import pickle
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union

# MONAI (Medical Open Network for AI) 라이브러리 구성요소 import 
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,                    # 여러 transform을 묶어 순차적으로 적용
    AddChanneld,                # channel dimension 추가
    RandCropByPosNegLabeld,     # label 양성/음성 비율 기반 랜덤 크롭
    CropForegroundd,            # 영상 내 foreground만 남기도록 크롭
    SpatialPadd,                # 지정된 크기보다 작을 경우 패딩
    ScaleIntensityRanged,       # intensity normalization (min-max scaling)
    RandShiftIntensityd,        # intensity 값 랜덤 이동
    RandFlipd,                  # 축 방향 랜덤 반전
    RandAffined,                # affine 변환 (회전, 이동, 확대 등)
    RandZoomd,                  # 랜덤 줌
    RandRotated,                # 랜덤 회전
    RandRotate90d,              # 90도 단위 랜덤 회전
    RandGaussianNoised,         # 랜덤 노이즈 추가
    RandGaussianSmoothd,        # 랜덤 가우시안 블러
    NormalizeIntensityd,        # 평균/표준편차 기반 정규화
    MapTransform,               # key 기반 커스텀 transform 상속용
    RandScaleIntensityd,        # intensity 랜덤 스케일 변환
    RandSpatialCropd,           # 공간적 랜덤 크롭
)
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib         # NIfTI 등 의료영상 입출력용
import torch.nn.functional as F


# ------------------------------------------------------------
# BinarizeLabeld: label(세그멘테이션 마스크)을 이진화하는 Transform
# ------------------------------------------------------------
class BinarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,          # 변환을 적용할 key 목록 (예: ["label"])
            threshold: float = 0.5,        # 임계값
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):  # 지정된 key들에 대해 순회
            if not isinstance(d[key], torch.Tensor):
                # numpy 배열일 경우 torch 텐서로 변환
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            # threshold 기준으로 0/1 이진화
            d[key] = (d[key] > self.threshold).to(dtype)
        return d


# ------------------------------------------------------------
# BaseVolumeDataset
#  - nibabel로 NIfTI 의료영상 읽기
#  - spacing(resampling), augmentation, normalization 처리
#  - MONAI transform 적용
# ------------------------------------------------------------
class BaseVolumeDataset(Dataset):
    def __init__(
            self,
            image_paths,                   # 이미지 경로 리스트
            label_meta,                    # 라벨 경로 리스트 (세그멘테이션 마스크)
            augmentation,                  # 데이터 증강 여부
            split="train",                 # 데이터 분할(train/val/test)
            rand_crop_spatial_size=(96, 96, 96),  # 랜덤 크롭 크기
            convert_to_sam=True,           # SAM 모델 호환 변환 여부
            do_test_crop=True,             # test 시 crop 수행 여부
            do_val_crop=True,              # validation 시 crop 수행 여부
            do_nnunet_intensity_aug=True,  # nnUNet 스타일 intensity 증강 여부
    ):
        super().__init__()
        # 경로 및 설정 저장
        self.img_dict = image_paths
        self.label_dict = label_meta
        self.aug = augmentation
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop

        # 데이터셋별 통계값 placeholder 
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None

        # 하위 클래스(KiTSVolumeDataset 등)에서 _set_dataset_stat()을 오버라이드
        self._set_dataset_stat()

        # transform pipeline 생성
        self.transforms = self.get_transforms()

    def _set_dataset_stat(self):
        """하위 클래스에서 intensity, spacing, target class 등을 정의"""
        pass

    def __len__(self):
        return len(self.img_dict)

    # ------------------------------------------------------------
    # 각 인덱스별 (이미지, 라벨) 페어 로드
    #  - nibabel로 NIfTI 읽기
    #  - spacing 보정
    #  - transform 적용
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        img_path = self.img_dict[idx]
        label_path = self.label_dict[idx]

        # -----------------------------
        # 1) NIfTI 파일 로드
        # -----------------------------
        img_vol = nib.load(img_path)
        img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)  # (H, W, D) 정렬
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])  # voxel spacing(mm)

        seg_vol = nib.load(label_path)
        seg = seg_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        # NaN 값은 0으로 대체
        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0

        # target class(예: 1 또는 2)에 해당하는 픽셀만 1로 설정
        seg = (seg == self.target_class).astype(np.float32)

        # -----------------------------
        # 2) 공간적 리샘플링 (spacing 보정)
        # -----------------------------
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or (
                np.max(self.target_spacing / np.min(self.target_spacing) > 8)
        ):
            # resize 2D
            # spacing 차이가 너무 큰 경우: 2D interpolation 방식 사용
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),  # (D,H,W) → (D,1,H,W)
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bilinear",
            )

            # 라벨은 test split이 아닐 때만 보간
            if self.split != "test":
                seg_tensor = F.interpolate(
                    input=torch.tensor(seg[:, None, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )

            # depth 방향 보간 (trilinear)
            img = (
                F.interpolate(
                    input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            # 일반적인 경우: 3D trilinear interpolation으로 spacing 맞춤
            img = (
                F.interpolate(
                    input=torch.tensor(img[None, None, :, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                    ),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, None, :, :, :]),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )

        # -----------------------------
        # 3) Transform (augmentation / crop 등)
        # -----------------------------
        if (self.aug and self.split == "train") or ((self.do_val_crop  and self.split=='val')):
            # 학습/검증 시 transform 적용
            trans_dict = self.transforms({"image": img, "label": seg})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        else:
            # test 시 transform만 적용 (aug 없이)
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        seg_aug = seg_aug.squeeze()  # 불필요한 채널 축 제거

        img_aug = img_aug.repeat(3, 1, 1, 1)  # 3채널로 확장 (SAM 등 RGB 입력 모델 호환용)

        # -----------------------------
        # 4) 반환 (입력, 라벨, spacing)
        # -----------------------------
        return img_aug, seg_aug, np.array(img_vol.header.get_zooms())[self.spatial_index]

    # ------------------------------------------------------------
    # get_transforms()
    # 데이터 split(train/val/test)에 따라 다른 MONAI transform pipeline 구성
    # ------------------------------------------------------------
    def get_transforms(self):
        # 기본 intensity scaling transform
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],
                b_max=self.intensity_range[1],
                clip=True,
            ),
        ]

        # -----------------------------
        # 학습용 transform 구성
        # -----------------------------
        if self.split == "train":
            transforms.extend(
                [
                    RandShiftIntensityd(  # 밝기 변동
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    CropForegroundd(      # foreground 기준으로 crop
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0],
                    ),
                    NormalizeIntensityd(  # 평균/표준편차 정규화
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )

            # 2D crop 기반 dataset의 경우 (예: pancreas, colon)
            if self.do_dummy_2D:
                transforms.extend(
                    [
                       RandRotated(   # 2D 회전
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                                ),
                        RandZoomd(    # 확대/축소
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )
            else:
                # 3D 회전 및 확대
                transforms.extend(
                    [
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            # 추가 crop/flip augmentation
            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(  # 패딩
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(  # 양성/음성 픽셀 기반 랜덤 크롭
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),
                    # 축 방향 랜덤 flip 및 90도 회전
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
                ]
            )

        # -----------------------------
        # 검증용 transform (val)
        # -----------------------------
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val"):
            transforms.extend(
                [
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )

        # -----------------------------
        # 테스트용 transform
        # -----------------------------
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        # 최종적으로 Compose로 묶어 반환
        transforms = Compose(transforms)

        return transforms