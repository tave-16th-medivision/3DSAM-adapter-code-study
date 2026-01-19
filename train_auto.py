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
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger

def main():
    # -------------------------------------------------------------------------
    # 1. 하이퍼파라미터 및 설정 파싱
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"],
        help="사용할 데이터셋 이름 선택"
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
        help="체크포인트 및 로그를 저장할 경로"
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
        help="데이터 파일이 위치한 경로 접두사"
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
        help="학습 시 무작위 크롭할 3D 패치 크기 (0이면 데이터셋별 기본값 사용)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="학습에 사용할 디바이스 (예: cuda:0)"
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int, help="분할할 클래스 개수")
    parser.add_argument("--lr", default=4e-4, type=float, help="학습률 (Learning Rate)")
    parser.add_argument("--max_epoch", default=500, type=int, help="최대 학습 에폭 수")
    parser.add_argument("--eval_interval", default=4, type=int, help="검증 수행 간격 (에폭 단위)")
    parser.add_argument("--resume", action="store_true", help="이전 학습 지점부터 재개 여부")
    parser.add_argument("--num_worker", default=6, type=int, help="데이터 로더 워커 수")
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()
    device = args.device

    # 데이터셋별 기본 크롭 사이즈 설정 (사용자가 지정하지 않은 경우)
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        # 입력된 크기가 1개면 3차원으로 확장, 아니면 그대로 사용
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    
    # 스냅샷 저장 경로 생성
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    # 로거 설정 (화면 출력 및 파일 저장)
    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    # -------------------------------------------------------------------------
    # 2. 데이터 로더 초기화
    # -------------------------------------------------------------------------
    # 학습 데이터 로더 (Augmentation 적용)
    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    # 검증 데이터 로더 (Augmentation 미적용, 결정론적 설정)
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
    # 3. 모델 초기화 및 Pretrained Weights 로드
    # -------------------------------------------------------------------------
    # SAM의 기본 모델(ViT-B) 로드 (사전 학습된 가중치 가져오기 위함)
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 3D 이미지 인코더 정의 (SAM의 2D 인코더를 3D로 확장한 버전으로 추정)
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

    # SAM 모델의 가중치를 3D 인코더로 복사 (strict=False로 3D 관련 추가 파라미터 제외하고 로드)
    img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    del sam  # 메모리 확보를 위해 원본 SAM 모델 삭제
    img_encoder.to(device)

    # -------------------------------------------------------------------------
    # 4. 파라미터 Freezing 및 Fine-tuning 설정
    # -------------------------------------------------------------------------
    # 기본적으로 인코더의 모든 파라미터를 학습되지 않도록 고정(Freeze)
    for p in img_encoder.parameters():
        p.requires_grad = False
    
    # 3D 처리를 위해 추가되거나 튜닝이 필요한 부분만 학습 가능하도록 설정(Unfreeze)
    img_encoder.depth_embed.requires_grad = True # 깊이 임베딩
    for p in img_encoder.slice_embed.parameters(): # 슬라이스 임베딩
        p.requires_grad = True
    
    # 각 블록 내의 Norm 레이어, Adapter 레이어, 위치 인코딩 등 미세 조정
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters(): # Adapter를 통해 3D 지식 학습
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        # 상대 위치 인코딩 조정
        i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
    
    # Neck 부분 학습 가능 설정
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    # 마스크 디코더 초기화 (MLAHead 사용)
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    mask_decoder.to(device)

    # -------------------------------------------------------------------------
    # 5. 옵티마이저 및 손실 함수 설정
    # -------------------------------------------------------------------------
    # 인코더와 디코더 각각 별도의 옵티마이저 및 스케줄러 설정
    # requires_grad=True인 파라미터만 필터링하여 옵티마이저에 전달
    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    # 검증용 Dice Loss (Gradient 계산 안 함)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    # 학습용 복합 Loss (Dice Loss + Cross Entropy Loss)
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]

    # -------------------------------------------------------------------------
    # 6. 학습 루프 (Training Loop)
    # -------------------------------------------------------------------------
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        mask_decoder.train()
        
        for idx, (img, seg, spacing) in enumerate(train_data):
            print('seg: ', seg.sum()) # 디버깅용: 세그멘테이션 레이블의 합 출력
            
            # 입력 이미지를 모델 입력 크기에 맞게 보간(Interpolation)
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            # 차원 변경: (B, C, D, H, W) -> (D, B, C, H, W) 형태 등으로 변환 추정
            input_batch = input_batch[0].transpose(0, 1)
            
            # 인코더 Forward Pass
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            new_feature = feature_list
            
            # 원본 이미지를 리사이즈하여 feature list에 추가 (Skip connection 역할)
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                mode='trilinear')
            new_feature.append(img_resize)
            
            # 디코더 Forward Pass
            masks = mask_decoder(new_feature, 2, patch_size//64)
            masks = masks.permute(0, 1, 4, 2, 3) # (B, C, D, H, W) 형태로 복원
            
            # Ground Truth 준비
            seg = seg.to(device)
            seg = seg.unsqueeze(1) # 채널 차원 추가
            
            # Loss 계산 및 역전파
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            loss.backward()
            
            # 로깅 및 Gradient Clipping
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 12.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 12.0)
            
            encoder_opt.step()
            decoder_opt.step()
            
        encoder_scheduler.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        # -------------------------------------------------------------------------
        # 7. 검증 루프 (Validation Loop)
        # -------------------------------------------------------------------------
        img_encoder.eval()
        mask_decoder.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing) in enumerate(val_data):
                print('seg: ', seg.sum())
                
                # 학습 시와 동일한 전처리 및 Forward Pass 수행
                out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
                input_batch = out.to(device)
                input_batch = input_batch[0].transpose(0, 1)
                
                batch_features, feature_list = img_encoder(input_batch)
                feature_list.append(batch_features)
                new_feature = feature_list
                
                img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                                           mode='trilinear')
                new_feature.append(img_resize)
                
                masks = mask_decoder(new_feature, 2, patch_size//64)
                masks = masks.permute(0, 1, 4, 2, 3)
                
                seg = seg.to(device)
                seg = seg.unsqueeze(1)
                
                # 검증 시에는 Dice Loss만 사용하여 평가
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
                        
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))

        # -------------------------------------------------------------------------
        # 8. 모델 저장 (Best Model Checkpoint)
        # -------------------------------------------------------------------------
        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        
        # 체크포인트 저장 (에폭, loss, 모델 가중치, 옵티마이저 상태 등)
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()