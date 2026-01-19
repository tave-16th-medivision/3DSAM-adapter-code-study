from dataset.datasets import load_data_volume
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
from monai.inferers import sliding_window_inference

def main():
    """
    3D 의료 영상 세그멘테이션 테스트 스크립트 (inference + 평가)
    - 저장된 encoder / decoder checkpoint를 불러와서
      테스트셋 전체에 대해 Dice, NSD를 계산하고 로그로 출력한다.
    """
    # ------------------------------
    # 1. ArgumentParser 설정
    # ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )

    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="last",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()

    # 사용할 checkpoint 파일 이름 설정 (last or best)
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    
    device = args.device

    # ------------------------------
    # 2. 입력 patch 크기 설정
    # ------------------------------
    # rand_crop_size가 0이면, 데이터셋 종류에 따라 기본값 (256^3) 사용
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        # 하나의 값만 들어온 경우 → (s,s,s)로 확장
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        # 세 값이 이미 주어진 경우 그대로 튜플화
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)

    # snapshot_path 하위에 data 이름을 붙여서 실험 디렉토리 구성
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    # ------------------------------
    # 3. Logger 설정
    # ------------------------------
    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))

    # ------------------------------
    # 4. 데이터 로더 구성 (test split)
    # ------------------------------
    test_data = load_data_volume(
        data=args.data,
        batch_size=1,                      # 테스트는 일반적으로 batch_size=1
        path_prefix=args.data_prefix,
        augmentation=False,                # 테스트시 augmentation 없음
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,                # 재현성을 위해 deterministic
        num_worker=0                       # 테스트용이라 worker 0
    )

    # ------------------------------
    # 5. Image Encoder 생성 및 weight 로드
    # ------------------------------
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    # encoder_dict 로드 (cpu로 로드 후 to(device))
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)
    img_encoder.to(device)


    # ------------------------------
    # 6. Mask Decoder 생성 및 weight 로드
    # ------------------------------
    mask_decoder = VIT_MLAHead(img_size = 96).to(device)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
    mask_decoder.to(device)

    # ------------------------------
    # 7. Loss 함수 및 eval 모드 설정
    # ------------------------------
    # DiceLoss: include_background=False → background 제외,
    # softmax=False → 이미 이진 마스크(or softmax 후 argmax), to_onehot_y=True → label one-hot 변환
    dice_loss = DiceLoss(
        include_background=False, 
        softmax=False, 
        to_onehot_y=True, 
        reduction="none"    # case별 loss를 개별적으로 보존
    )
    img_encoder.eval()
    mask_decoder.eval()

    # patch_size: 현재 구현에서는 cubic patch 가정, 첫 번째 축만 사용
    patch_size = args.rand_crop_size[0]


    # ------------------------------
    # 8. sliding window 내에서 사용하는 예측 함수 정의
    # ------------------------------
    def model_predict(img, img_encoder, mask_decoder):
        """
        sliding_window_inference에서 사용할 predictor 함수.
        Args:
            img: 입력 이미지 텐서 (B, C, D, H, W)
        Returns:
            masks: softmax 후 foreground 클래스 채널만 남긴 예측 (B, 1, D, H, W) 형태
        """
        # 입력 영상을 encoder가 기대하는 해상도(512)로 업샘플
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')

        # encoder 입력 형태에 맞게 차원 변환
        # out: (B, C, D, H, W) → out[0]: (C, D, H, W) → (D, C, H, W) 형태로 transpose
        input_batch = out[0].transpose(0, 1)

        # ViT 기반 3D encoder를 통해 멀티스케일 feature 추출
        batch_features, feature_list = img_encoder(input_batch)
        # 마지막 global feature도 리스트 맨 뒤에 추가
        feature_list.append(batch_features)

        new_feature = feature_list

        # 원본 intensity volume(또는 single channel)을 decoder에 추가로 제공
        # img[0,0]: (D, H, W) → (H, W, D)로 permute 후, (1,1,D',H',W')로 reshape & 업샘플
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size,
                                   mode="trilinear")
        new_feature.append(img_resize)

        # mask decoder를 통해 segmentation logit 예측
        masks = mask_decoder(new_feature, 2, patch_size//64)
        # decoder 출력 차원을 (B, C, D, H, W) 형태로 재배열
        masks = masks.permute(0, 1, 4, 2, 3)
        # softmax로 class별 확률로 변환
        masks = torch.softmax(masks, dim=1)
        # background 채널 제외 (foreground만 사용)
        masks = masks[:, 1:]
        return masks

    # ------------------------------
    # 9. Evaluation 루프
    # ------------------------------
    with torch.no_grad():
        loss_summary = []  # case별 Dice score 모음
        loss_nsd = []      # case별 NSD score 모음
        for idx, (img, seg, spacing) in enumerate(test_data):
            # seg: (C, D, H, W) 형태의 label volume
            seg = seg.float()
            seg = seg.to(device)
            img = img.to(device)

            # sliding window inference:
            # - roi_size: [256,256,256]
            # - overlap: 0.5
            # - gaussian window blending
            pred = sliding_window_inference(img, [256, 256, 256], overlap=0.5, sw_batch_size=1,
                                            mode="gaussian",
                                            predictor=partial(model_predict,
                                                              img_encoder=img_encoder,
                                                              mask_decoder=mask_decoder))
            
            # 예측 결과를 GT(seg) 해상도에 맞게 다시 trilinear 업샘플
            pred = F.interpolate(pred, size=seg.shape[1:], mode="trilinear")
            # seg: (C, D, H, W) → (1, C, D, H, W)로 batch 차원 추가
            seg = seg.unsqueeze(0)

            # --------------------------
            # 특별 케이스:
            # pred가 거의 전부 background이고,
            # seg도 전부 background인 경우 → Dice, NSD = 1로 처리
            # --------------------------
            if torch.max(pred) < 0.5 and torch.max(seg) == 0:
                loss_summary.append(1)
                loss_nsd.append(1)
            else:
                # 0.5 threshold로 binary mask 생성
                masks = pred > 0.5

                # DiceLoss는 "1 - Dice" 형태이므로, Dice score를 얻으려면 1 - loss
                loss = 1 - dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                # surface distance 기반 NSD 계산
                ssd = surface_distance.compute_surface_distances(
                    (seg == 1)[0, 0].cpu().numpy(),     # GT binary mask
                    (masks == 1)[0, 0].cpu().numpy(),   # Pred binary mask
                    spacing_mm=spacing[0].numpy()       # voxel spacing
                )
                # tolerance(mm) 내에서의 surface dice
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
                loss_nsd.append(nsd)

            # 각 case별 결과 로그
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.img_dict[idx], loss.item(), nsd
                ))
            
        # ------------------------------
        # 10. 전체 테스트셋 평균 지표 출력
        # ------------------------------
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()

