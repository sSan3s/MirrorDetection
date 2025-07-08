import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


# MViTv2 구성 요소들
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, kernel_q=(1, 1), kernel_kv=(1, 1),
                 stride_q=(1, 1), stride_kv=(1, 1), qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5

        self.conv_q = nn.Conv2d(dim, dim_out, kernel_q, stride_q, padding=(kernel_q[0] // 2, kernel_q[1] // 2))
        self.conv_k = nn.Conv2d(dim, dim_out, kernel_kv, stride_kv, padding=(kernel_kv[0] // 2, kernel_kv[1] // 2))
        self.conv_v = nn.Conv2d(dim, dim_out, kernel_kv, stride_kv, padding=(kernel_kv[0] // 2, kernel_kv[1] // 2))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape

        # Multi-scale queries, keys, values
        q = self.conv_q(x)  # B, dim_out, H', W'
        k = self.conv_k(x)  # B, dim_out, H'', W''
        v = self.conv_v(x)  # B, dim_out, H'', W''

        # 출력 크기 가져오기
        _, _, Hq, Wq = q.shape
        _, _, Hkv, Wkv = k.shape

        # Reshape for attention
        q = q.reshape(B, self.num_heads, self.head_dim, Hq * Wq).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, Hkv * Wkv).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, Hkv * Wkv).transpose(-2, -1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v)  # B, num_heads, Hq*Wq, head_dim

        # Reshape back
        x = x.transpose(-2, -1).reshape(B, self.dim_out, Hq, Wq)

        # Final projection
        x = x.permute(0, 2, 3, 1).contiguous().view(B * Hq * Wq, self.dim_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, Hq, Wq, self.dim_out).permute(0, 3, 1, 2)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., kernel_q=(1, 1), kernel_kv=(1, 1),
                 stride_q=(1, 1), stride_kv=(1, 1)):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = nn.LayerNorm(dim)

        self.attn = MultiScaleAttention(
            dim, dim_out, num_heads=num_heads, qkv_bias=qkv_bias,
            kernel_q=kernel_q, kernel_kv=kernel_kv,
            stride_q=stride_q, stride_kv=stride_kv,
            drop=drop, attn_drop=attn_drop
        )

        self.norm2 = nn.LayerNorm(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, out_features=dim_out, drop=drop)

        # Downsampling 또는 차원 변경을 위한 projection
        if dim != dim_out or stride_q[0] > 1:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride_q)
        else:
            self.proj = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Norm + Attention
        x_norm = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x_norm = self.norm1(x_norm).view(B, H, W, C).permute(0, 3, 1, 2)
        x_attn = self.attn(x_norm)

        # Residual connection with optional projection
        if self.proj is not None:
            x = self.proj(x)
            # 출력 크기 업데이트 (stride로 인한 크기 변경)
            _, _, H, W = x.shape
        x = x + x_attn

        # Norm + MLP
        x_norm = x.permute(0, 2, 3, 1).contiguous().view(-1, self.dim_out)
        x_norm = self.norm2(x_norm)
        x_mlp = self.mlp(x_norm).view(B, H, W, self.dim_out).permute(0, 3, 1, 2)
        x = x + x_mlp

        return x


# MViTv2 백본 - 수정된 버전
class MViTv2Backbone(nn.Module):
    def __init__(self, img_size=224, patch_size=7, in_chans=3, embed_dim=96,
                 depth=[2, 3, 16, 3], num_heads=[1, 2, 4, 8], mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_stages = len(depth)

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=4, padding=patch_size // 2)

        # Multi-scale transformer blocks
        dims = [embed_dim]
        for i in range(1, self.num_stages):
            dims.append(dims[-1] * 2)

        self.blocks = nn.ModuleList()
        current_dim = embed_dim  # 현재 차원을 추적

        for stage_idx in range(self.num_stages):
            stage_blocks = []
            stage_dim = dims[stage_idx]

            for block_idx in range(depth[stage_idx]):
                # 각 stage의 첫 번째 블록에서 차원 변경 및 downsampling
                if stage_idx > 0 and block_idx == 0:
                    # Stage 전환 시: 이전 stage의 출력 차원 -> 현재 stage의 차원
                    block = MultiScaleBlock(
                        dim=current_dim,  # 이전 stage의 출력 차원
                        dim_out=stage_dim,  # 현재 stage의 차원
                        num_heads=num_heads[stage_idx],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        kernel_q=(1, 1),
                        kernel_kv=(3, 3),
                        stride_q=(2, 2),  # Downsampling
                        stride_kv=(2, 2)
                    )
                    current_dim = stage_dim
                else:
                    # 같은 stage 내의 블록들: 차원 유지
                    block = MultiScaleBlock(
                        dim=current_dim,
                        dim_out=current_dim,
                        num_heads=num_heads[stage_idx],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        kernel_q=(1, 1),
                        kernel_kv=(3, 3),
                        stride_q=(1, 1),
                        stride_kv=(1, 1)
                    )

                stage_blocks.append(block)

            self.blocks.append(nn.Sequential(*stage_blocks))

        self.dims = dims  # 각 stage의 출력 차원 저장
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Extract multi-scale features
        features = []
        for stage_idx, stage in enumerate(self.blocks):
            x = stage(x)
            features.append(x)

        return features


# HetNet의 MIC 모듈 (Multi-orientation Intensity-based Contrasted)
class MIC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MIC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3x1 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv1x3 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        # Multi-orientation convolutions
        x1 = self.conv3x3(x)
        x2 = self.conv3x1(x)
        x3 = self.conv1x3(x)
        x = x1 + x2 + x3
        x = self.bn(x)
        x = self.relu(x)
        return x


# HetNet의 RSL 모듈 (Reflection Semantic Logical)
class RSL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        # Multi-scale receptive fields
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x = x1 + x2
        # Channel attention
        se = self.se(x)
        x = x * se
        x = self.bn(x)
        x = self.relu(x)
        return x


# 완전한 HetNet with MViTv2
class HetNetMViTv2(nn.Module):
    def __init__(self, num_classes=1):
        super(HetNetMViTv2, self).__init__()

        # MViTv2 백본
        self.backbone = MViTv2Backbone(
            embed_dim=96,
            depth=[2, 3, 16, 3],  # MViT-B configuration
            num_heads=[1, 2, 4, 8],  # 5를 4로 변경 (384는 4로 나누어떨어짐)
            mlp_ratio=4.
        )

        # Feature dimensions from MViTv2: [96, 192, 384, 768]
        dims = self.backbone.dims

        # Low-level feature processing (MIC modules)
        self.mic1 = MIC(dims[0], 64)
        self.mic2 = MIC(dims[1], 128)

        # High-level feature processing (RSL modules)
        self.rsl3 = RSL(dims[2], 256)
        self.rsl4 = RSL(dims[3], 512)

        # Feature fusion
        self.fusion_low = nn.Sequential(
            nn.Conv2d(64 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.fusion_high = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Final fusion and prediction
        self.final_fusion = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        # Extract multi-scale features from MViTv2
        features = self.backbone(x)

        # Process low-level features
        low1 = self.mic1(features[0])
        low2 = self.mic2(features[1])

        # Process high-level features
        high1 = self.rsl3(features[2])
        high2 = self.rsl4(features[3])

        # Resize features to match dimensions
        size = low1.shape[2:]
        low2 = F.interpolate(low2, size=size, mode='bilinear', align_corners=False)
        high1 = F.interpolate(high1, size=size, mode='bilinear', align_corners=False)
        high2 = F.interpolate(high2, size=size, mode='bilinear', align_corners=False)

        # Fuse features
        low_fused = self.fusion_low(torch.cat([low1, low2], dim=1))
        high_fused = self.fusion_high(torch.cat([high1, high2], dim=1))

        # Final prediction
        final = self.final_fusion(torch.cat([low_fused, high_fused], dim=1))

        # Upsample to original resolution
        final = F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=False)

        return final


# 모델 초기화 및 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = HetNetMViTv2(num_classes=1)

    # 입력 예시
    input_tensor = torch.randn(2, 3, 512, 640)

    # Forward pass
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 각 stage의 출력 확인
    print("\nFeature shapes at each stage:")
    features = model.backbone(input_tensor)
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape}")