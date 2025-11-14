import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAHead(nn.Module):
    """
    Multi-Level Aggregation Head (3D 버전)
    - backbone / ViT 등에서 나온 4단계 feature map (mla_p2~mla_p5)을
      각각 3D conv block으로 처리한 뒤, 동일 해상도로 업샘플해서 채널 방향으로 concat.
    - 최종적으로 "다단계 특징을 하나의 풍부한 feature volume"으로 만드는 역할.
    """
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        # 각 단계별 feature에 대해 동일 구조의 3D conv block 정의
        # 입력 채널: mla_channels (예: 256), 출력 채널: mlahead_channels (예: 128)

        # stage 2 feature용 head
        self.head2 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        # stage 3 feature용 head
        self.head3 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        # stage 4 feature용 head
        self.head4 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        # stage 5 feature용 head
        self.head5 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5, scale_factor):
        """
        Args:
            mla_p2, mla_p3, mla_p4, mla_p5:
                서로 다른 해상도/스케일의 feature map (B, C=mla_channels, D, H, W)
            scale_factor:
                최종 출력 해상도까지 trilinear 업샘플할 비율 (float or tuple)

        Returns:
            torch.Tensor:
                [head2, head3, head4, head5]를 채널 방향(dim=1)으로 concat한 텐서
                shape: (B, 4 * mlahead_channels, D', H', W')
        """
        # 각 단계 feature에 대해 3D conv block을 통과시키고 동일 해상도까지 업샘플
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        
        # 다단계 feature를 channel 방향으로 concat
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    Vision Transformer + MLA Head (3D segmentation)
    - ViT 등의 backbone에서 나온 multi-level feature, 그리고 추가 1채널 입력(예: 원본 또는 mask)을 받아
      최종 segmentation logit을 출력하는 3D 헤드.
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, num_classes=3,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size            # 기준 이미지 크기 (정방형, cubic 가정)
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        # multi-level feature들을 모아주는 3D MLA head
        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        
        # 최종 segmentation classifier head
        # 입력 채널: 4 * mlahead_channels (multi-level concat) + 1 (추가 입력)
        self.cls = nn.Sequential(nn.Conv3d(4 * mlahead_channels + 1, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 3, padding=1, bias=False))

    def forward(self, inputs, scale_factor=None):
        """
        Args:
            inputs:
                list/tuple 길이 ≥ 5 가정
                inputs[0..3]: multi-level feature (mla_p2~mla_p5), shape: (B, mla_channels, D, H, W)
                inputs[-1]:   추가 1채널 입력 (예: 원본 intensity / mask 등), shape: (B, 1, D', H', W')
                              → MLA 출력과 동일한 해상도여야 concat 가능
            scale_factor:
                None이면, inputs[0]의 spatial size를 img_size에 맞추는 비율로 자동 계산
                (img_size / inputs[0].shape[-1]) : cubic volume 가정(D=H=W)

        Returns:
            torch.Tensor:
                segmentation logit, shape: (B, num_classes, D_out, H_out, W_out)
        """
        # scale_factor가 지정되지 않았다면,
        # 첫 번째 feature map의 마지막 dimension 기준으로 자동 설정
        if scale_factor == None:
            scale_factor = self.img_size / inputs[0].shape[-1]

        # MLA Head로 multi-level feature를 합치고 업샘플
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3], scale_factor = scale_factor)

        # multi-level feature concat 결과에 추가 1채널 입력을 함께 concat
        # (B, 4*mlahead_channels, D,H,W) + (B,1,D,H,W) → (B,4*mlahead_channels+1,D,H,W)
        x = torch.cat([x, inputs[-1]], dim=1)

        # 최종 segmentation logit 계산
        x = self.cls(x)
        return x


class VIT_MLAHead_h(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    Vision Transformer + MLA Head (hierarchical upsampling 버전)
    - 위 VIT_MLAHead와 거의 동일하지만,
      마지막에 한 번 더 trilinear 업샘플(scale_factor2)하여 더 높은 해상도의 예측을 반환.
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, num_classes=2,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead_h, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        # multi-level feature aggregation head
        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        
        # classifier (num_classes만 다름: 예를 들어 binary segmentation)
        self.cls = nn.Sequential(nn.Conv3d(4 * mlahead_channels + 1, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 3, padding=1, bias=False))

    def forward(self, inputs, scale_factor1, scale_factor2):
        """
        Args:
            inputs:
                inputs[0..3]: multi-level feature maps
                inputs[-1]:   추가 1채널 입력
            scale_factor1:
                MLA head 내부에서 각 stage feature를 공통 해상도로 올릴 때 사용
            scale_factor2:
                cls 출력(로그릿)을 최종 target 해상도로 한 번 더 업샘플할 때 사용

        Returns:
            torch.Tensor:
                최종 업샘플된 segmentation logit
        """
        # 1단계: multi-level feature를 공통 해상도까지 업샘플 및 concat
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3], scale_factor = scale_factor1)

        # 추가 1채널 입력과 concat
        x = torch.cat([x, inputs[-1]], dim=1)

        # segmentation logit 계산
        x = self.cls(x)

        # 2단계: 더 높은 해상도로 최종 업샘플
        x = F.interpolate(x, scale_factor = scale_factor2, mode='trilinear', align_corners=True)
        return x