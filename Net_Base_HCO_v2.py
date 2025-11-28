import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import math

from resnext.resnext101_regular import ResNeXt101


############################################ Initialization ##############################################
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) or isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax) or isinstance(m,
                                                                                                              nn.Sigmoid) or isinstance(
            m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.ReLU6):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            m.initialize()


############################################ Basic ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


############################################ Deformable Convolution ##############################################
class DeformableConv2dWithOffset(nn.Module):
    """Deformable Convolution with learnable offset"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConv2dWithOffset, self).__init__()

        # Offset convolution (learns 2*kernel_size*kernel_size offsets)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 2 for x,y offsets per kernel position
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )

        # Deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Initialize offset conv to zero (start with regular convolution)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        # Learn offsets
        offsets = self.offset_conv(x)
        # Apply deformable convolution
        out = self.deform_conv(x, offsets)
        return out

    def initialize(self):
        # Re-initialize offset conv to zeros after weight_init
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)


############################################ DCT/IDCT Implementation (Compatible with PyTorch < 1.12) ##############################################
class DCT2D(nn.Module):
    """
    2D Discrete Cosine Transform - Compatible with older PyTorch versions
    Implements DCT using matrix multiplication for gradient support
    """

    def __init__(self):
        super(DCT2D, self).__init__()
        self.dct_h_cache = {}
        self.dct_w_cache = {}

    def get_dct_matrix(self, N, device, dtype):
        """Generate DCT matrix"""
        key = (N, device, dtype)
        if key not in self.dct_h_cache:
            n = torch.arange(N, dtype=dtype, device=device)
            k = n.view(-1, 1)

            # DCT-II matrix
            dct_matrix = torch.cos(math.pi / N * (n + 0.5) * k)
            # Normalization
            dct_matrix[0, :] *= math.sqrt(1.0 / N)
            dct_matrix[1:, :] *= math.sqrt(2.0 / N)

            self.dct_h_cache[key] = dct_matrix

        return self.dct_h_cache[key]

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W) in frequency domain
        """
        B, C, H, W = x.shape

        # Get DCT matrices
        dct_h = self.get_dct_matrix(H, x.device, x.dtype)  # (H, H)
        dct_w = self.get_dct_matrix(W, x.device, x.dtype)  # (W, W)

        # Reshape for batch matrix multiplication
        x_reshaped = x.reshape(B * C, H, W)

        # Apply DCT along height: X_freq = dct_h @ X
        x_dct_h = torch.matmul(dct_h, x_reshaped)  # (B*C, H, W)

        # Apply DCT along width: X_freq = X_freq @ dct_w^T
        x_dct_hw = torch.matmul(x_dct_h, dct_w.t())  # (B*C, H, W)

        # Reshape back
        x_freq = x_dct_hw.reshape(B, C, H, W)

        return x_freq


class IDCT2D(nn.Module):
    """
    2D Inverse Discrete Cosine Transform - Compatible with older PyTorch versions
    """

    def __init__(self):
        super(IDCT2D, self).__init__()
        self.idct_h_cache = {}
        self.idct_w_cache = {}

    def get_idct_matrix(self, N, device, dtype):
        """Generate IDCT matrix (transpose of normalized DCT matrix)"""
        key = (N, device, dtype)
        if key not in self.idct_h_cache:
            n = torch.arange(N, dtype=dtype, device=device)
            k = n.view(-1, 1)

            # DCT-II matrix
            dct_matrix = torch.cos(math.pi / N * (n + 0.5) * k)
            # Normalization
            dct_matrix[0, :] *= math.sqrt(1.0 / N)
            dct_matrix[1:, :] *= math.sqrt(2.0 / N)

            # IDCT is transpose of DCT
            idct_matrix = dct_matrix.t()

            self.idct_h_cache[key] = idct_matrix

        return self.idct_h_cache[key]

    def forward(self, x):
        """
        x: (B, C, H, W) in frequency domain
        returns: (B, C, H, W) in spatial domain
        """
        B, C, H, W = x.shape

        # Get IDCT matrices
        idct_h = self.get_idct_matrix(H, x.device, x.dtype)  # (H, H)
        idct_w = self.get_idct_matrix(W, x.device, x.dtype)  # (W, W)

        # Reshape for batch matrix multiplication
        x_reshaped = x.reshape(B * C, H, W)

        # Apply IDCT along height: X_spatial = idct_h @ X_freq
        x_idct_h = torch.matmul(idct_h, x_reshaped)  # (B*C, H, W)

        # Apply IDCT along width: X_spatial = X_spatial @ idct_w^T
        x_idct_hw = torch.matmul(x_idct_h, idct_w.t())  # (B*C, H, W)

        # Reshape back
        x_spatial = x_idct_hw.reshape(B, C, H, W)

        return x_spatial


############################################ vHeat HCO Components ##############################################
class FrequencyValueEmbedding(nn.Module):
    """
    Frequency Value Embeddings (FVEs) for thermal diffusivity prediction
    ✅ FIXED: Now uses spatial frequency embeddings like original vHeat
    """

    def __init__(self, dim, num_heads=1, resolution=None):
        super(FrequencyValueEmbedding, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution  # Expected (H, W)

        # ✅ NEW: Spatial frequency embeddings (H, W, C) like vHeat original
        # If resolution is not given, will be initialized during first forward pass
        if resolution is not None:
            H, W = resolution
            # Shape: (H, W, C) - permuted to (C, H, W) for PyTorch convention
            self.freq_embed = nn.Parameter(torch.zeros(dim, H, W))
            nn.init.trunc_normal_(self.freq_embed, std=0.02)
        else:
            self.freq_embed = None

        # Linear layer to predict thermal diffusivity k
        self.k_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_heads)
        )

    def forward(self, x_freq, h, w):
        """
        x_freq: (B, C, H, W) frequency domain features
        Returns: thermal diffusivity k for heat conduction
        """
        B, C = x_freq.shape[0], x_freq.shape[1]

        # Initialize freq_embed if not done yet (for dynamic resolution)
        if self.freq_embed is None or self.freq_embed.shape[1:] != (h, w):
            self.freq_embed = nn.Parameter(torch.zeros(C, h, w, device=x_freq.device, dtype=x_freq.dtype))
            nn.init.trunc_normal_(self.freq_embed, std=0.02)

        # ✅ NEW: Add spatial frequency embeddings
        # freq_embed: (C, H, W) -> expand to (1, C, H, W)
        freq_embed_expanded = self.freq_embed.unsqueeze(0)  # (1, C, H, W)

        # Interpolate if size mismatch (for flexibility)
        if freq_embed_expanded.shape[2:] != x_freq.shape[2:]:
            freq_embed_expanded = F.interpolate(freq_embed_expanded, size=(h, w),
                                                mode='bilinear', align_corners=True)

        # Add FVE to frequency features
        x_with_fve = x_freq + freq_embed_expanded  # (B, C, H, W)

        # Global average pooling to predict k
        fve_pooled = F.adaptive_avg_pool2d(x_with_fve, (1, 1))  # (B, C, 1, 1)
        fve_pooled = fve_pooled.view(B, C)  # (B, C)

        k = self.k_predictor(fve_pooled)  # (B, num_heads)

        return k

    def initialize(self):
        if self.freq_embed is not None:
            nn.init.trunc_normal_(self.freq_embed, std=0.02)
        for m in self.k_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class HeatConductionOperator(nn.Module):
    """Heat Conduction Operator (HCO) - Core module of vHeat with numerical stability"""

    def __init__(self, dim, num_heads=1, t=1.0, resolution=None):
        super(HeatConductionOperator, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.t = t  # conduction time (fixed constant)

        self.dct = DCT2D()
        self.idct = IDCT2D()
        self.fve = FrequencyValueEmbedding(dim, num_heads, resolution)

        # Linear projection for input
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        With numerical stability improvements
        """
        B, C, H, W = x.shape

        # Linear projection
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
        x_proj = self.proj_in(x_flat)  # (B*H*W, C)
        x_proj = x_proj.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Transform to frequency domain
        x_freq = self.dct(x_proj)  # (B, C, H, W)

        # Check for NaN/Inf after DCT
        if torch.isnan(x_freq).any() or torch.isinf(x_freq).any():
            x_freq = torch.nan_to_num(x_freq, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ Predict thermal diffusivity k using FVEs (now with spatial embeddings)
        k = self.fve(x_freq, H, W)  # (B, num_heads)

        # Clamp k to prevent extreme values (k should be positive and reasonable)
        k = torch.clamp(k, min=0.01, max=10.0)

        # Create frequency grid (ω_x, ω_y) with normalization to prevent overflow
        freq_x = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1) / H
        freq_y = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W) / W
        omega_squared = (freq_x ** 2 + freq_y ** 2)  # (1, 1, H, W)

        # Heat diffusion: exp(-k * ω^2 * t)
        # Average k across heads if multiple heads
        k_mean = k.mean(dim=1, keepdim=True).view(B, 1, 1, 1)  # (B, 1, 1, 1)

        # Clamp exponent to prevent overflow/underflow
        # exp(-10) ≈ 0.000045, exp(0) = 1
        exponent = -k_mean * omega_squared * self.t
        exponent = torch.clamp(exponent, min=-10.0, max=0.0)  # Prevent positive exponents

        diffusion_factor = torch.exp(exponent)  # (B, 1, H, W)

        # Apply heat diffusion in frequency domain
        x_diffused = x_freq * diffusion_factor  # (B, C, H, W)

        # Check for NaN/Inf after diffusion
        if torch.isnan(x_diffused).any() or torch.isinf(x_diffused).any():
            x_diffused = x_freq  # Fallback to input if diffusion causes issues

        # Transform back to spatial domain
        x_spatial = self.idct(x_diffused)  # (B, C, H, W)

        # Check for NaN/Inf after IDCT
        if torch.isnan(x_spatial).any() or torch.isinf(x_spatial).any():
            x_spatial = torch.nan_to_num(x_spatial, nan=0.0, posinf=0.0, neginf=0.0)

        # Output projection
        x_spatial_flat = x_spatial.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x_out = self.proj_out(x_spatial_flat)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return x_out

    def initialize(self):
        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)
        self.fve.initialize()


class HCOLayer(nn.Module):
    """
    Complete HCO Layer (similar to Transformer block)
    Structure: DWConv -> (HCO branch + Gating branch) -> FFN
    """

    def __init__(self, dim, num_heads=1, mlp_ratio=4.0, drop=0., t=1.0, resolution=None):
        super(HCOLayer, self).__init__()
        self.dim = dim

        # Depth-wise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)

        # HCO branch (with resolution info)
        self.hco = HeatConductionOperator(dim, num_heads, t, resolution)

        # Gating branch (multiplicative gating signal)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Save for residual
        identity = x

        # Depth-wise convolution
        x = self.dwconv(x)

        # Apply layer norm (need to permute for LayerNorm)
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)

        # HCO branch
        x_hco = self.hco(x_norm)  # (B, C, H, W)

        # Gating branch
        x_gate_flat = x_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)
        gate_signal = self.gate(x_gate_flat)  # (B*H*W, C)
        gate_signal = gate_signal.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Apply gating
        x = x_hco * gate_signal

        # Residual connection
        x = identity + x

        # FFN with residual
        identity2 = x
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = identity2 + x

        return x

    def initialize(self):
        nn.init.trunc_normal_(self.dwconv.weight, std=0.02)
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        self.hco.initialize()
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


############################################ Feature Fusion Module ##############################################
# RepConv block for advanced FFM
class rep_conv(nn.Module):
    def __init__(self, channel):
        super(rep_conv, self).__init__()
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.norm0 = nn.BatchNorm2d(channel)
        self.norm1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        identity = x
        x_3x3 = self.conv3x3(x)
        x_1x1 = self.conv1x1(x)
        return F.relu(identity + self.norm0(x_3x3) + self.norm1(x_1x1))

    def initialize(self):
        weight_init(self)


# Advanced Feature Fusion Module with RepConv
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1)
        self.conv1x1_rep = nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1)
        self.rep_block = nn.ModuleList([rep_conv(channel) for _ in range(7)])

    def forward(self, x_1, x_2):
        cat = torch.cat([x_1, x_2], dim=1)
        x_1x1 = self.conv1x1(cat)
        x_rep = self.conv1x1_rep(cat)

        for rep in self.rep_block:
            x_rep = rep(x_rep)

        return x_1x1 + x_rep

    def initialize(self):
        weight_init(self)


############################################ Cross Aggregation Module ##############################################
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)


####################################### Reflection Semantic Logical Module with DCN ##########################################
class RFB_modified(nn.Module):
    '''RSL with Deformable 7x7 Convolution - reflection semantic logical module'''

    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )

        # Branch 1: 1x1 → Deformable 7x7 → 3x3 (d=7)
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            DeformableConv2dWithOffset(out_channel, out_channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )

        # Branch 2: 1x1 → Deformable 7x7 → Deformable 7x7 → 3x3 (d=7)
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            DeformableConv2dWithOffset(out_channel, out_channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            DeformableConv2dWithOffset(out_channel, out_channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )

        self.conv_cat = basicConv(3 * out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

    def initialize(self):
        weight_init(self)
        for module in self.modules():
            if isinstance(module, DeformableConv2dWithOffset):
                module.initialize()


############################################## CoordAtt & SAM #############################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        pass


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        pass


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

    def initialize(self):
        weight_init(self)


class SAM(nn.Module):
    def __init__(self, nin: int, nout: int, num_splits: int) -> None:
        super(SAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [CoordAtt(int(self.nin / self.num_splits), int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out

    def initialize(self):
        weight_init(self)


class IntensityPositionModule(nn.Module):
    '''multi-orientation intensity-based contrasted module (MIC)'''

    def __init__(self, inplanes, outplanes, g=1):
        super(IntensityPositionModule, self).__init__()
        self.SA1 = SAM(inplanes, outplanes, g)
        self.SA2 = SAM(inplanes, outplanes, g)

        self.conv = nn.Sequential(
            basicConv(inplanes, inplanes, k=3, s=1, p=1, d=1, g=inplanes),
            basicConv(inplanes, outplanes, k=1, s=1, p=0, relu=True)
        )

    def forward(self, x):
        y = x.clone()

        y = torch.rot90(y, 1, dims=[2, 3])
        y = self.SA1(y)
        y = torch.rot90(y, -1, dims=[2, 3])

        x = self.SA2(x)

        out = x * y
        out = self.conv(out)

        return out

    def initialize(self):
        weight_init(self)


############################################## Pooling #############################################
class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)


##################################### Main Network with HCO Pathway - FVE Fixed ###################################################
class Net(nn.Module):
    def __init__(self, cfg, backbone_path="./resnext/resnext_101_32x4d.pth"):
        super(Net, self).__init__()
        self.cfg = cfg

        # Original ResNeXt101 backbone (kept intact)
        self.bkbone = ResNeXt101(backbone_path)

        # HCO Pathway depth configuration
        self.HCO_depth = [2, 2, 9, 2]

        # ✅ Expected resolutions for each stage (approximate for 352x352 input)
        # Adjust these based on your actual feature map sizes
        # Stage1: H/4, Stage2: H/8, Stage3: H/16, Stage4: H/32, Stage5: H/32
        resolutions = [
            (176, 176),  # stage1: 352/4
            (88, 88),  # stage2: 352/8
            (44, 44),  # stage3: 352/16
            (22, 22),  # stage4: 352/32
            (11, 11)  # stage5: 352/32
        ]

        # HCO Pathway with resolution info for FVE
        self.hco_pathway = nn.ModuleList([
            nn.Sequential(*[HCOLayer(dim=64, num_heads=1, mlp_ratio=4.0, t=1.0, resolution=resolutions[0])
                            for _ in range(self.HCO_depth[0])]),
            nn.Sequential(*[HCOLayer(dim=256, num_heads=4, mlp_ratio=4.0, t=1.0, resolution=resolutions[1])
                            for _ in range(self.HCO_depth[1])]),
            nn.Sequential(*[HCOLayer(dim=512, num_heads=8, mlp_ratio=4.0, t=1.0, resolution=resolutions[2])
                            for _ in range(self.HCO_depth[2])]),
            nn.Sequential(*[HCOLayer(dim=1024, num_heads=16, mlp_ratio=4.0, t=1.0, resolution=resolutions[3])
                            for _ in range(self.HCO_depth[3])]),
            HCOLayer(dim=2048, num_heads=32, mlp_ratio=4.0, t=1.0, resolution=resolutions[4])
        ])

        # ✅ Simple Concat + Conv1x1 fusion (instead of FFM)
        self.hco_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 + 256, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512 + 512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024 + 1024, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(2048 + 2048, 2048, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True)
            )
        ])

        # Channel alignment layers for HCO pathway connections
        self.hco_align = nn.ModuleList([
            nn.Conv2d(64, 256, 1),
            nn.Conv2d(256, 512, 1),
            nn.Conv2d(512, 1024, 1),
            nn.Conv2d(1024, 2048, 1),
        ])

        # Replace Pyramid Pooling (GE) with projection from final HCO output
        self.hco_to_64 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.ModuleList([
            basicConv(64, 64, k=1, s=1, p=0),
            basicConv(256, 64, k=1, s=1, p=0),
            basicConv(512, 64, k=1, s=1, p=0),
            basicConv(1024, 64, k=1, s=1, p=0),
            basicConv(2048, 64, k=1, s=1, p=0),
            basicConv(2048, 64, k=1, s=1, p=0)
        ])

        self.head = nn.ModuleList([
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
        ])

        self.ffm = nn.ModuleList([
            FFM(64),
            FFM(64),
            FFM(64),
            FFM(64),
            FFM(64)
        ])

        self.ipm = nn.ModuleList([
            IntensityPositionModule(64, 64),
            IntensityPositionModule(64, 64),
            IntensityPositionModule(64, 64)
        ])

        self.cam = CAM(64)
        self.ca1 = RFB_modified(1024, 64)
        self.ca2 = RFB_modified(2048, 64)
        self.refine = basicConv(64, 64, k=1, s=1, p=0)

        self.initialize()

    def forward(self, x, shape=None, return_intermediates=False):
        shape = x.size()[2:] if shape is None else shape

        # ===== ResNeXt101 Backbone =====
        bk_stage1, bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)

        # ===== HCO Pathway (원래대로 복원) =====
        # Stage 1: 독립적으로 처리
        hco_out1 = self.hco_pathway[0](bk_stage1)

        # Stage 2: Concat + Conv1x1
        hco1_aligned = self.hco_align[0](hco_out1)
        hco1_aligned = F.interpolate(hco1_aligned, size=bk_stage2.size()[2:],
                                     mode='bilinear', align_corners=True)
        hco_in2 = torch.cat([bk_stage2, hco1_aligned], dim=1)
        hco_in2 = self.hco_fusion[0](hco_in2)
        hco_out2 = self.hco_pathway[1](hco_in2)

        # Stage 3: Concat + Conv1x1
        hco2_aligned = self.hco_align[1](hco_out2)
        hco2_aligned = F.interpolate(hco2_aligned, size=bk_stage3.size()[2:],
                                     mode='bilinear', align_corners=True)
        hco_in3 = torch.cat([bk_stage3, hco2_aligned], dim=1)
        hco_in3 = self.hco_fusion[1](hco_in3)
        hco_out3 = self.hco_pathway[2](hco_in3)

        # Stage 4: Concat + Conv1x1
        hco3_aligned = self.hco_align[2](hco_out3)
        hco3_aligned = F.interpolate(hco3_aligned, size=bk_stage4.size()[2:],
                                     mode='bilinear', align_corners=True)
        hco_in4 = torch.cat([bk_stage4, hco3_aligned], dim=1)
        hco_in4 = self.hco_fusion[2](hco_in4)
        hco_out4 = self.hco_pathway[3](hco_in4)

        # Stage 5: Concat + Conv1x1
        hco4_aligned = self.hco_align[3](hco_out4)
        hco4_aligned = F.interpolate(hco4_aligned, size=bk_stage5.size()[2:],
                                     mode='bilinear', align_corners=True)
        hco_in5 = torch.cat([bk_stage5, hco4_aligned], dim=1)
        hco_in5 = self.hco_fusion[3](hco_in5)
        hco_out5 = self.hco_pathway[4](hco_in5)

        # ===== Replace GE with HCO output =====
        fused4 = self.hco_to_64(hco_out5)

        # ===== Rest of the network remains the same =====
        f5 = self.ca2(bk_stage5)
        fused4 = F.interpolate(fused4, size=f5.size()[2:], mode='bilinear', align_corners=True)
        fused3 = self.ffm[4](f5, fused4)

        f4 = self.ca1(bk_stage4)
        fused3 = F.interpolate(fused3, size=f4.size()[2:], mode='bilinear', align_corners=True)
        fused2 = self.ffm[3](f4, fused3)

        f3 = self.conv1[2](bk_stage3)
        f3 = self.ipm[2](f3)

        f2 = self.conv1[1](bk_stage2)
        f2 = self.ipm[1](f2)
        f3 = F.interpolate(f3, size=f2.size()[2:], mode='bilinear', align_corners=True)
        fused1 = self.ffm[2](f2, f3)

        fused2 = F.interpolate(fused2, size=[fused1.size(2) // 2, fused1.size(3) // 2],
                               mode='bilinear', align_corners=True)

        fused1 = self.cam(fused2, fused1)

        f1 = self.conv1[0](bk_stage1)
        f1 = self.ipm[0](f1)
        f2 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        fused0 = self.ffm[1](f2, f1)

        fused1 = F.interpolate(fused1, size=fused0.size()[2:], mode='bilinear', align_corners=True)
        out0 = self.ffm[0](fused1, fused0)

        out0 = self.refine(out0)

        edge0 = F.interpolate(self.head[0](fused0), size=shape, mode='bilinear', align_corners=True)
        main_out = F.interpolate(self.head[1](out0), size=shape, mode='bilinear', align_corners=True)

        out1 = F.interpolate(self.head[2](fused1), size=shape, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head[3](fused2), size=shape, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.head[4](fused3), size=shape, mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.head[5](fused4), size=shape, mode='bilinear', align_corners=True)

        if self.cfg.mode == 'train':
            return main_out, edge0, out1, out2, out3, out4
        elif return_intermediates:
            return main_out, edge0, out1, out2, out3, out4
        else:
            return main_out, edge0

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)