"""
이 코드는 3DSAM-adapter 모델을 학습시키기 위한 스크립트입니다.
주요 기능은 다음과 같습니다:
1. argparse를 사용하여 명령줄 인수 파싱
2. 지정된 데이터셋 로드 및 전처리
3. 사전 학습된 SAM 모델에서 이미지 인코더 초기화
4. 3D 이미지 인코더, 프롬프트 인코더, 마스크 디코더 정의
5. 옵티마이저 및 학습률 스케줄러 설정
6. 지정된 에포크 수 동안 모델 학습
7. 검증 데이터셋을 사용하여 주기적으로 모델 평가
8. 최적의 모델 체크포인트 저장
"""

from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger

def main():
    # -------------------------------------------------------------------------
    # 1. 설정 및 하이퍼파라미터 파싱
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"],
        help="사용할 데이터셋 이름 (예: kits, pancreas 등)"
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
        help="학습된 모델 체크포인트와 로그를 저장할 경로"
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
        help="데이터셋 파일이 위치한 루트 경로"
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
        help="학습 시 무작위 크롭할 3D 패치 크기. 0이면 데이터셋별 기본값 사용"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="학습에 사용할 디바이스 설정"
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float, help="학습률 (Learning Rate)")
    parser.add_argument("--max_epoch", default=500, type=int, help="최대 학습 에폭 수")
    parser.add_argument("--eval_interval", default=4, type=int, help="검증 수행 간격")
    parser.add_argument("--resume", action="store_true", help="중단된 학습 재개 여부")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()
    device = args.device
    
    # 데이터셋별 기본 크롭 사이즈 설정 (사용자가 지정하지 않은 경우)
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    
    # 스냅샷 저장 경로 생성
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    # 로거 설정
    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    # -------------------------------------------------------------------------
    # 2. 데이터 로더 초기화
    # -------------------------------------------------------------------------
    # 학습용 데이터 로더: Augmentation 적용
    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    # 검증용 데이터 로더: 결정론적(Deterministic) 설정, Augmentation 미적용
    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )

    # -------------------------------------------------------------------------
    # 3. 모델 초기화 및 가중치 이식
    # -------------------------------------------------------------------------
    # SAM의 기본 모델(ViT-B) 로드 (사전 학습된 가중치 활용 목적)
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 3D 이미지 인코더 정의 (SAM의 구조를 3D로 확장)
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

    # 사전 학습된 SAM의 가중치를 3D 인코더로 로드 (strict=False로 형태가 다른 부분은 제외)
    img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    del sam # 메모리 절약을 위해 원본 SAM 모델 삭제
    img_encoder.to(device)

    # -------------------------------------------------------------------------
    # 4. 파라미터 Freezing 및 Fine-tuning 설정 (PEFT)
    # -------------------------------------------------------------------------
    # 인코더의 모든 파라미터를 Freeze
    for p in img_encoder.parameters():
        p.requires_grad = False
    
    # 3D 적응을 위해 필요한 부분만 Unfreeze (학습 가능하도록 설정)
    img_encoder.depth_embed.requires_grad = True # 깊이 임베딩
    for p in img_encoder.slice_embed.parameters(): # 슬라이스 임베딩
        p.requires_grad = True
    
    # 각 블록 내의 Adapter 및 Norm 레이어 학습 허용
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters(): # Adapter 모듈
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        # 상대 위치 인코딩 조정
        i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
    
    # Neck 부분 학습 허용
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    # -------------------------------------------------------------------------
    # 5. 프롬프트 인코더 및 디코더 설정
    # -------------------------------------------------------------------------
    # 여러 레벨의 피처를 처리하기 위해 Prompt Encoder 리스트 생성
    prompt_encoder_list = []
    parameter_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                                                                 embedding_dim=256,
                                                                 mlp_dim=2048,
                                                                 num_heads=8))
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)
        # 학습 가능한 파라미터 수집
        parameter_list.extend([i for i in prompt_encoder.parameters() if i.requires_grad == True])

    # 마스크 디코더 초기화
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    mask_decoder.to(device)

    # -------------------------------------------------------------------------
    # 6. 옵티마이저 및 손실 함수 설정
    # -------------------------------------------------------------------------
    # 인코더, 프롬프트 인코더(feature), 디코더 각각 별도의 옵티마이저 설정
    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    feature_opt = AdamW(parameter_list, lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=500)
    
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    # 손실 함수 정의: 검증용(Dice) 및 학습용(Dice + CrossEntropy)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]

    # -------------------------------------------------------------------------
    # 7. 학습 루프 (Training Loop)
    # -------------------------------------------------------------------------
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        for module in prompt_encoder_list:
            module.train()
        mask_decoder.train()
        
        for idx, (img, seg, spacing) in enumerate(train_data):
            print('seg: ', seg.sum()) # 디버깅용: 마스크 픽셀 수 출력
            
            # 입력 이미지 리사이징 및 GPU 업로드
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1) # 차원 변경
            
            # 이미지 인코더 실행
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            
            # -----------------------------------------------------------------
            # 포인트 프롬프트 시뮬레이션 (Interactive Segmentation 학습용)
            # -----------------------------------------------------------------
            l = len(torch.where(seg == 1)[0]) # 전경(Foreground) 픽셀 수 확인
            points_torch = None
            
            # 전경에서 랜덤 포인트 샘플링 (Positive Points)
            if l > 0:
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch = points.to(device)
                points_torch = points_torch.transpose(0,1)
            
            # 배경에서 랜덤 포인트 샘플링 (Negative Points)
            l = len(torch.where(seg < 10)[0])
            sample = np.random.choice(np.arange(l), 20, replace=True)
            x = torch.where(seg < 10)[1][sample].unsqueeze(1)
            y = torch.where(seg < 10)[3][sample].unsqueeze(1)
            z = torch.where(seg < 10)[2][sample].unsqueeze(1)
            points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
            points_torch_negative = points.to(device)
            points_torch_negative = points_torch_negative.transpose(0, 1)
            
            # Positive와 Negative 포인트 결합
            if points_torch is not None:
                points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
            else:
                points_torch = points_torch_negative
            
            # -----------------------------------------------------------------
            # 프롬프트 인코딩 및 마스크 디코딩
            # -----------------------------------------------------------------
            new_feature = []
            for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                # 특정 레이어(여기서는 마지막 인덱스 3)에만 포인트 프롬프트 적용
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                    )
                else:
                    new_feature.append(feature)
            
            # 원본 이미지를 리사이즈하여 feature list에 추가 (Skip connection 역할)
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                mode='trilinear')
            new_feature.append(img_resize)
            
            # 마스크 예측
            masks = mask_decoder(new_feature, 2, patch_size//64)
            masks = masks.permute(0, 1, 4, 2, 3) # (B, C, D, H, W) 형태로 복원
            
            # Ground Truth 준비 및 Loss 계산
            seg = seg.to(device)
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            
            # 역전파 및 가중치 업데이트
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()
            
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder_list[-1].parameters(), 1.0)
            
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
            
        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        # -------------------------------------------------------------------------
        # 8. 검증 루프 (Validation Loop)
        # -------------------------------------------------------------------------
        img_encoder.eval()
        for module in prompt_encoder_list:
            module.eval()
        mask_decoder.eval()
        
        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing) in enumerate(val_data):
                print('seg: ', seg.sum())
                # 검증 데이터에 대해 학습과 동일한 전처리 및 포인트 샘플링 수행
                out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
                input_batch = out.to(device)
                input_batch = input_batch[0].transpose(0, 1)
                
                batch_features, feature_list = img_encoder(input_batch)
                feature_list.append(batch_features)
                
                # 포인트 샘플링 (검증 시에도 모델의 프롬프트 반응성을 평가하기 위함)
                l = len(torch.where(seg == 1)[0])
                points_torch = None
                if l > 0:
                    sample = np.random.choice(np.arange(l), 10, replace=True)
                    x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                    y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                    z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                    points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                    points_torch = points.to(device)
                    points_torch = points_torch.transpose(0, 1)
                
                l = len(torch.where(seg < 10)[0])
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg < 10)[1][sample].unsqueeze(1)
                y = torch.where(seg < 10)[3][sample].unsqueeze(1)
                z = torch.where(seg < 10)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch_negative = points.to(device)
                points_torch_negative = points_torch_negative.transpose(0, 1)
                
                if points_torch is not None:
                    points_torch = points_torch # 여기서는 concat 로직이 약간 다름 (수정 불가가 원칙이라 설명만 추가: 원본 코드 의도 확인 필요)
                else:
                    points_torch = points_torch_negative
                
                new_feature = []
                for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                    if i == 3:
                        new_feature.append(
                            prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                        )
                    else:
                        new_feature.append(feature)
                
                img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                                           mode='trilinear')
                new_feature.append(img_resize)
                
                masks = mask_decoder(new_feature, 2, patch_size//64)
                masks = masks.permute(0, 1, 4, 2, 3)
                seg = seg.to(device)
                seg = seg.unsqueeze(1)
                
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))

        # -------------------------------------------------------------------------
        # 9. 모델 체크포인트 저장
        # -------------------------------------------------------------------------
        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": [i.state_dict() for i in prompt_encoder_list],
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()