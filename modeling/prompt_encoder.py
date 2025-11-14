import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Type
import math
from segment_anything.modeling.common import MLPBlock


# -----------------------------------------------------------
# 3D Feature Normalization (Channel-wise LayerNorm for 3D)
# -----------------------------------------------------------
class LayerNorm3d(nn.Module):
    """
    LayerNorm in 3D, normalizing over channel dimension.
    input: B x C x D x H x W
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean over channel dimension
        u = x.mean(1, keepdim=True)
        # variance
        s = (x - u).pow(2).mean(1, keepdim=True)
        # normalize
        x = (x - u) / torch.sqrt(s + self.eps)
        # affine transform
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


# -----------------------------------------------------------
# Simple Adapter Block (residual MLP)
# -----------------------------------------------------------
class Adapter(nn.Module):
    """
    Used to inject low-rank transformation into features.
    """
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim, hidden_dim=mid_dim, output_dim=input_dim, num_layers=2
        )

    def forward(self, features):
        # Residual MLP Adapter
        out = features + self.model(features)
        return out


# -----------------------------------------------------------
# Simple 2-layer MLP (used in Adapter)
# -----------------------------------------------------------
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    """
    Linear -> ReLU -> Linear MLP block
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        # Construct Linear stacks
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # Apply each layer
        for i, layer in enumerate(self.layers):
            # 마지막을 제외한 모든 layer는 ReLU 적용
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# -----------------------------------------------------------
# Two-Way Transformer (SAM 구조)
# - 이미지 토큰과 sparse 토큰(point/global)을 교차-attention
# -----------------------------------------------------------
class TwoWayTransformer(nn.Module):
    """
    SAM-style transformer for prompt conditioning.
    image_embedding: (B, C, H, W)
    global queries + point queries → iterative transformer updates
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()


        # Build multi-layer transformer blocks
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_coord,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C


        # -------------------------
        # 1. Extract sparse point embedding via grid_sample
        #    point_coord: B×N×3 (normalized to [-1,1])
        # -------------------------
        point_embedding = F.grid_sample(image_embedding, point_coord, align_corners=False).squeeze(2).squeeze(2)
        
        # same for positional encoding
        point_pe = F.grid_sample(image_pe, point_coord, align_corners=False).squeeze(2).squeeze(2)

        # Rearrange B×C×N → B×N×C
        point_pe = point_pe.permute(0, 2, 1)
        point_embedding = point_embedding.permute(0, 2, 1)
        original_shape = image_embedding.shape

        # -------------------------
        # 2. Flatten image embedding B×C×H×W → B×(HW)×C
        # -------------------------
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # -------------------------
        # 3. Apply multi-layer transformer blocks
        # -------------------------
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            image_embedding, point_embedding = layer(
                image_embedding,
                point_embedding,
                image_pe,
                point_pe,
            )
        return image_embedding


# -----------------------------------------------------------
# A single Two-Way attention block
# (SAM의 핵심 구조)
# 1) self-attn on (global + point)
# 2) sparse→dense cross attention
# 3) MLP
# 4) dense→sparse cross attention
# -----------------------------------------------------------
class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()

        # Self-attention on token queries (global + points)
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Token → Image cross-attention
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # MLP for token update
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Image → Token cross-attention
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        # --------- Global Learnable Query ---------
        # 10개의 learnable global tokens (SAM의 output token 역할)
        self.global_query = nn.parameter.Parameter(data=0.1 * torch.randn(1, 10, embedding_dim))

    def forward(self, img_embed, point_embed, img_pe, point_pe) -> Tuple[Tensor, Tensor]:
        # -------------------------------------------------------
        # 1. Concatenate global tokens + point tokens
        #    q shape: B×(10+N_points)×C
        # -------------------------------------------------------
        q = torch.cat([self.global_query, point_embed], dim=1)

        # ---------------- Self-attention ----------------
        self_out = self.self_attn(q=q, k=q, v=q)
        self_out = self.norm1(self_out)

        # Cross attention block, tokens attending to image embedding
        # residual + norm
        queries = q + self_out
        queries = self.norm2(queries)
        # split tokens:
        # global tokens = first 10
        # point tokens = rest
        point_embed = queries[:, 10:, :]
        queries = queries[:, :10, :]

        # ---------------- MLP update (only global queries) ----------------
        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # ---------------- Image → Global Token cross attention ----------------
        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_image_to_token(q=img_embed, k=queries, v=queries)
        keys = img_embed + attn_out
        # update image tokens
        keys = self.norm4(keys)

        return keys, point_embed


# -----------------------------------------------------------
# Generic Multi-Head Attention Module
# with optional downsample_rate
# -----------------------------------------------------------
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    -- Standard multi-head attention with optional channel downsampling.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()

        # Projection to smaller dim if downsample_rate > 1
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        # Q,K,V projection
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        # B×N×C → B×H×N×(C/H)
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        # B×H×N×D → B×N×(H*D)
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Project to Q,K,V
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # attention scores B×H×N×N
        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        # attention apply
        out = attn @ v
        # recombine heads + final projection
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# -----------------------------------------------------------
# Prompt Encoder (3D points → embeddings)
# - point_coord → grid_sample → token
# -----------------------------------------------------------
class PromptEncoder(nn.Module):
    """
    Encode 3D point prompts into embeddings + apply TwoWayTransformer.
    """
    def __init__(
        self,
        *,
        transformer: nn.Module,
        num_pos_feats: int = 128,
        mask_prompt = False
    ) -> None:
        super().__init__()
        self.transformer = transformer

        # Gaussian matrix for positional encoding
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
             torch.randn((3, num_pos_feats)),
        )
        self.mask_prompt = mask_prompt
        if mask_prompt:
            # default 3D mask embedding
            self.default_prompt = nn.parameter.Parameter(torch.randn(1, 256, 32, 32, 32))
            # 3D mask encoder: Conv3D → Norm → Conv3D → ...
            self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 256 // 4, kernel_size=3, stride=3),
            LayerNorm3d(256 // 4),
            nn.GELU(),
            nn.Conv3d(256 // 4, 256, kernel_size=3, padding = 1, stride=1),
            LayerNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, 256, kernel_size=1),
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coord,
        img_size = [512, 512, 32],
        feat_size = [32, 32, 32]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # ---------------------------------------
        # 1. Positional Encoding for image grid
        # ---------------------------------------
        image_pe = self.get_img_pe(feat_size, device=image_embeddings.device).detach()
        '''
        if self.mask_prompt:
            if masks == None:
                image_embeddings += self.default_prompt
            else:
                image_embeddings += self.mask_encoder(masks)
        '''

        # ---------------------------------------
        # 2. Convert pixel coords → normalized [-1,1]
        #    grid_sample expects normalized coords
        # ---------------------------------------
        point_coord[:, :, 0] = (point_coord[:, :, 0]+0.5) * 2 / img_size[2] - 1
        point_coord[:, :, 1] = (point_coord[:, :, 1]+0.5) * 2 / img_size[1] - 1
        point_coord[:, :, 2] = (point_coord[:, :, 2]+0.5) * 2 / img_size[0] - 1
        # reshape to B×1×1×N×3 for 3D grid_sample
        point_coord = point_coord.reshape(1,1,1,-1,3)

        # ---------------------------------------
        # 3. Apply TwoWayTransformer
        # ---------------------------------------
        features = self.transformer(image_embeddings, image_pe, point_coord)
        # B×(HW)×C → reshape back to B×C×D×H×W
        features = features.transpose(1,2).reshape([1, -1] + feat_size)

        return features
    
    # -------------------------------------------------------
    # Positional Encoding for 3D coords
    # -------------------------------------------------------
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points that are normalized to [0,1].
        coords normalized to [0,1], shape: D×H×W×3
        Produces sinusoidal encoding using Gaussian matrix.
        """
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1    # to [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords * 3 / 2
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0,1].
        Alternative path: directly encode normalized coords
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def get_img_pe(self, size: Tuple[int, int], device) -> torch.Tensor:
        """
        Generate positional encoding for a grid of the specified size.
        Build 3D positional encoding grid for image embedding.
        size: (D, H, W)
        output: B×C×D×H×W
        """
        h, w, d = size
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2).unsqueeze(0)  # C x D X H x W     # B×C×D×H×W
