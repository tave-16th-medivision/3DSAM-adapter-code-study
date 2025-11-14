import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Attention, PatchEmbed, window_partition, window_unpartition

class Adapter(nn.Module):
    """
    3D ViT 블록 내부에 삽입되는 어댑터 모듈.
    - 입력 특성 차원을 mid_dim으로 줄였다가 다시 복원하는 bottleneck 구조
    - 중간에서 depthwise 3D conv로 지역적인 3D 컨텍스트를 학습
    - 마지막에 입력과 residual connection으로 더해줌
    """
    def __init__(
            self,
            input_dim,
            mid_dim
    ):
        super().__init__()
        # 채널 축에 대해 선형 변환 (C -> mid_dim)
        self.linear1 = nn.Linear(input_dim, mid_dim)
        # depthwise 3D conv: 채널 별로 독립적인 3D convolution
        self.conv = nn.Conv3d(in_channels = mid_dim, out_channels = mid_dim, kernel_size=3, padding=1, groups=mid_dim)
        # 다시 원래 채널 차원으로 복원 (mid_dim -> C)
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features):
        # features: [B, H, W, D, C] 형태를 가정
        out = self.linear1(features)        # 채널 축 FC
        out = F.relu(out)
        # Conv3d 입력을 위해 채널을 두 번째 차원으로 이동: [B, C_mid, H, W, D]
        out = out.permute(0, 4, 1, 2, 3)
        out = self.conv(out)
        # 다시 [B, H, W, D, C_mid]로 되돌리기
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)             # 채널 복원
        out = F.relu(out)
        # residual 연결: 입력 + 어댑터 출력
        out = features + out
        return out

class LayerNorm3d(nn.Module):
    """
    채널 차원 기준으로 3D 텐서에 LayerNorm을 적용하는 모듈.
    - 입력: [B, C, D, H, W]
    - 채널(C)마다 weight, bias를 학습
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 채널 차원(C)을 기준으로 평균/분산 계산
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # 채널별 스케일/시프트 적용
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class ImageEncoderViT_3d(nn.Module):
    """
    2D SAM ImageEncoderViT를 3D 볼륨에 맞게 확장한 버전 (v1).
    - 2D PatchEmbed로 slice 단위 패치 추출
    - num_slice > 1 인 경우 Conv3d를 이용해 슬라이스 방향으로 downsample
    - 절대 위치 임베딩 (H, W, D 방향) 추가
    - Block_3d를 depth 만큼 쌓아서 3D ViT backbone 구성
    - neck_3d를 통해 중간/최종 feature를 out_chans 채널로 투영
    """
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int=32,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice = 1
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        # 2D Patch Embedding: (H, W) 방향으로 patch_size 간격의 패치 생성
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_slice = num_slice
        if self.num_slice > 1:
            # 여러 슬라이스를 하나의 3D voxel로 묶기 위한 depthwise Conv3d
            self.slice_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                         kernel_size=(1,1,self.num_slice), stride=(1,1,self.num_slice),
                                         groups=embed_dim)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # 2D 위치 임베딩 (H/patch_size, W/patch_size, C)
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            # Depth 방향 위치 임베딩 (D, C)
            self.depth_embed = nn.Parameter(
                torch.zeros(1, patch_depth, embed_dim)
            )

        # ViT 블록 리스트 (3D 버전)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,  # 3D window attention 크기
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0  # 짝수 block에 대해서는 shifted window
            )
            self.blocks.append(block)

        # neck_3d: 중간/최종 feature를 out_chans로 매핑하는 projection layer
        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(nn.Sequential(
                nn.Conv3d(768, out_chans, 1, bias=False),
                nn.InstanceNorm3d(out_chans),
                nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Patch Embedding
        # SAM pretrained 가중치를 그대로 사용하기 위해 patch_embed는 grad를 막고 사용
        with torch.no_grad():
            x = self.patch_embed(x)  # 원래 [B, C, H, W] -> [B, H', W', C]

        # (2) Slice Embedding (num_slice > 1 인 경우)
        if self.num_slice > 1:
            # [H', W', B, C] -> [1, C, H', W', B] 형태로 바꿔서 Conv3d에 넣음
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            # 다시 [1, H', W', D', C] 형태로
            x = x.permute(0, 2, 3, 4, 1)
        else:
            # 단일 슬라이스인 경우: [B, H', W', C] -> [1, H', W', B, C] (D 차원에 B를 매핑)
            x = x.permute(1, 2, 0, 3).unsqueeze(0)

        # (3) 3D 절대 위치 임베딩 추가
        # 절대 위치 임베딩 추가 (H, W, D 세 방향)
        if self.pos_embed is not None:
            # pos_embed는 원래 pretrain 해상도 기준이기 때문에 avg_pool2d로 다운샘플
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
            # depth_embed와 broadcast해서 3D 위치 정보 구성
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            x = x + pos_embed

        # (4) 3D ViT 블록 통과
        idx = 0
        feature_list = []
        # 앞 6개 블록
        for blk in self.blocks[:6]:
            x = blk(x)
            idx += 1
            # 3블록마다 neck를 통과시켜 중간 feature를 추출 (마지막 블록 제외)
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))
        # 뒤 6개 블록
        for blk in self.blocks[6:12]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))

        # (5) 3D ViT 블록 통과 후, 최종 neck_3d 적용
        # 마지막 블록 출력도 neck를 통과시켜 최종 feature 추출
        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        # x: 최종 feature (1, C_out, H', W', D'), feature_list: multi-scale 3D feature map들
        return x, feature_list

class ImageEncoderViT_3d_v2(nn.Module):
    """
    ImageEncoderViT_3d의 변형 버전 (v2).
    - neck_3d에 InstanceNorm 대신 LayerNorm3d + 3D Conv를 사용 (더 깊은 local 처리)
    - depth_embed를 ones로 초기화 (v1은 zeros)
    """
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int=32,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice = 1
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        # 2D Patch Embedding은 v1과 동일
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_slice = num_slice
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                         kernel_size=(1,1,self.num_slice), stride=(1,1,self.num_slice),
                                         groups=embed_dim)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            # v1과 다르게 ones로 초기화 (초기 depth 방향 bias를 부여)
            self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            )

        # Block_3d 구성은 v1과 동일
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0
            )
            self.blocks.append(block)

        # v2 neck_3d: Conv1x1 -> LayerNorm3d -> Conv3x3 -> LayerNorm3d
        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(nn.Sequential(
                nn.Conv3d(768, out_chans, 1, bias=False),
                LayerNorm3d(out_chans),
                nn.Conv3d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm3d(out_chans),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Patch Embedding
        with torch.no_grad():
            x = self.patch_embed(x)

        # (2) Slice Embedding (num_slice > 1 인 경우)
        if self.num_slice > 1:
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)

        # (3) 3D 절대 위치 임베딩 추가
        if self.pos_embed is not None:
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            x = x + pos_embed

        # (4) 3D ViT 블록 통과
        idx = 0
        feature_list = []
        for blk in self.blocks[:6]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                # 3블록마다 neck를 통과시켜 중간 feature를 추출 (마지막 블록 제외)
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))
        for blk in self.blocks[6:12]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))

        # (5) 3D ViT 블록 통과 후, 최종 neck_3d 적용
        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))

        return x, feature_list

class Block_3d(nn.Module):
    """
    3D Transformer Block.
    - 3D window attention (shifted window까지 지원)
    - Adapter 모듈을 사용한 경량 residual branch
    - MLPBlock를 통한 채널-wise 확장
    """
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            res_size = None,
            shift = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 3D Multi-head Attention
        self.attn = Attention_3d(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(window_size, window_size, window_size),
            res_size=(res_size, res_size, res_size),
        )
        self.shift_size = shift
        if self.shift_size > 0:
            # Shifted window attention용 attention mask를 미리 생성 (shape 기반)
            H, W, D = 32, 32, 32   # 마스크를 위한 고정 템플릿 크기 (실제 입력에 대해 리사이즈 역할)
            img_mask = torch.zeros((1, H, W, D, 1))
            # Swin 방식과 유사한 3D 영역 분할
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            d_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
            # window 단위로 mask를 partition
            mask_windows = window_partition(img_mask, window_size)[0]
            mask_windows = mask_windows.view(-1, window_size * window_size * window_size)
            # 서로 다른 영역끼리는 attention을 막기 위해 큰 음수(-100)로 마스킹
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
                                                                                         float(0.0))
        else:
            attn_mask = None
        # 학습 파라미터로 저장 (forward에서 자동으로 device 이동)
        self.register_buffer("attn_mask", attn_mask)


        self.norm2 = norm_layer(dim)
        # MLPBlock: 채널-wise fully-connected FFN
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
        # Adapter: attention 전단에 삽입되는 residual 모듈
        self.adapter = Adapter(input_dim=dim, mid_dim=dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapter를 먼저 적용 (입력에 local 3D conv 기반 residual 추가)
        x = self.adapter(x)
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]
            # Shifted window를 위해 공간 축을 롤링
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1,2,3))
            x, pad_hw = window_partition(x, self.window_size)
        # 3D Attention 수행
        x = self.attn(x, mask=self.attn_mask)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W, D))
        if self.shift_size > 0:
            # 다시 원래 위치로 롤백
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1,2,3))

        # 첫 번째 residual (Self-Attention)
        x = shortcut + x
        # 두 번째 residual (MLP)
        x = x + self.mlp(self.norm2(x))
        return x

class Attention_3d(nn.Module):
    """
    3D Multi-head Self-Attention 블록.
    - 입력: [B, H, W, D, C]
    - qkv projection 후 head별 attention 계산
    - 필요 시 3D relative positional bias 추가
    """
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            res_size = None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5   # scaled dot-product attention에서의 scaling factor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # 높이, 너비, 깊이 방향 각각의 relative position embedding
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * res_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * res_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * res_size[2] - 1, head_dim))
            # relative position bias의 스케일을 학습하는 파라미터
            self.lr = nn.Parameter(torch.tensor(1.))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, H, W, D, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        # qkv: [3, B, num_heads, HW D, head_dim]
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        # q, k, v: [B, num_heads, HW D, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        #q, k, v = qkv.reshape(3, B * self.num_heads, H * W * D, -1).unbind(0)
        # relative position 계산을 위해 [B * num_heads, HW D, head_dim]로 reshape
        q_sub = q.reshape(B * self.num_heads, H * W * D, -1)

        # scaled dot-product attention (마스크 적용 전)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            # 3D decomposed relative positional bias 추가
            attn = add_decomposed_rel_pos(attn, q_sub, self.rel_pos_h, self.rel_pos_w, self.rel_pos_d, (H, W, D), (H, W, D), self.lr)
            attn = attn.reshape(B, self.num_heads, H * W * D, -1)
        if mask is None:
            attn = attn.softmax(dim=-1)
        else:
            # shifted window용 attention mask 적용
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, H*W*D, H*W*D) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, H*W*D, H*W*D)
            attn = attn.softmax(dim=-1)
        # [B, num_heads, H, W, D, head_dim] -> [B, H, W, D, C]
        x = (attn @ v).view(B, self.num_heads, H, W, D, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, D, -1)
        x = self.proj(x)

        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    3D 윈도우 단위로 패딩 후 분할하는 함수.
    Args:
        x (tensor): input tokens with [B, H, W, C].  -- [B, H, W, D, C]?? 
        window_size (int): window size.  -- 윈도우 한 변의 길이
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
                    -- (Hp, Wp, Dp): 패딩 후의 H, W, D 크기
    """
    B, H, W, D, C = x.shape
    
    # 크기가 window_size로 나누어떨어지지 않으면 패딩
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        # F.pad의 인자 순서: (C, C, D, D, W, W, H, H)
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    # [B, Hp/window, window, Wp/window, window, Dp/window, window, C] 형태로 나눈 뒤
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    # 윈도우 단위를 배치 차원으로 모으기
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Hp, Wp, Dp)

def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int, int], hw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    window_partition의 역연산: 윈도우들을 다시 원래 3D feature map으로 합치고 패딩 제거.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
                        -- (Hp, Wp, Dp) 패딩 후 높이/너비/깊이
        hw (Tuple): original height and width (H, W) before padding.
                        -- (H, W, D) 패딩 전 원래 높이/너비/깊이
    Returns:
        x: unpartitioned sequences with [B, H, W, C].  -- x: [B, H, W, D, C]
    """
    Hp, Wp, Dp = pad_hw
    H, W, D = hw
    B = windows.shape[0] // (Hp * Wp * Dp // window_size // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    # 처음에 붙인 패딩 부분 제거
    if Hp > H or Wp > W or Dp > D:
        x = x[:, :H, :W, :D, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    query와 key의 길이에 맞게 relative positional embedding을 추출/보간하는 함수.
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q. -- query의 길이
        k_size (int): size of key k. -- key의 길이
        rel_pos (Tensor): relative position embeddings (L, C).
                            -- [L, C] 형태의 relative pos embedding 테이블
    Returns:
        Extracted positional embeddings according to relative positions.
        -- relative position에 맞게 indexing된 [q_size*k_size, C] 형태의 embedding
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # 필요 시 1D linear interpolation으로 테이블 크기 보간
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # q, k의 위치 차이를 기반으로 테이블 index 계산
    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        rel_pos_d: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
        lr,
) -> torch.Tensor:
    """
    mvitv2 논문에서 제안된 decomposed 3D relative positional embedding을 attention에 추가.
    - H, W, D 축의 relative bias를 각각 계산 후 합산.
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map. -- [B, N, N] 형태의 attention map
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C). -- [B, N, C] 형태의 query
        -- rel_pos_h, rel_pos_w, rel_pos_d: 각 축에 대한 rel pos embedding 테이블
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis. 
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        -- q_size, k_size: (H, W, D) 형태의 query/key 공간 크기
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
        -- lr: relative bias의 스케일을 학습하는 파라미터
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
                -- relative positional bias가 추가된 attention map
    """
    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size
    # 각 축에 대해 크기 맞춰 보간된 rel pos 테이블을 얻음
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)

    B, _, dim = q.shape
    # q를 [B, H, W, D, C]로 reshape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    # 각 축의 relative bias 계산
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)

    # attn: [B, H, W, D, H, W, D] 형태로 reshape 후 각 축의 bias를 더해줌
    attn = (
            attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d) +
            lr * rel_h[:, :, :, :, :, None, None] +
            lr * rel_w[:, :, :, :, None, :, None] +
            lr * rel_d[:, :, :, :, None, None, :]
    ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn
