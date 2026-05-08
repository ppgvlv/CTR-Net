import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np

from model import resnet18
from model.unet import UNetAnySize


# ================= 2D RoPE =================
def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """初始化 Mixed RoPE 的随机频率"""
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    """计算 Mixed RoPE 的复数旋转因子"""
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """计算 Axial RoPE 的复数旋转因子"""
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def init_t_xy(end_x: int, end_y: int):
    """初始化 2D 网格坐标"""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整 freqs_cis 形状以便广播"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb_2d(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """应用 2D 旋转位置编码到 q 和 k"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ================= DropPath =================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ================= 多头注意力机制（带 2D RoPE + mask） =================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_2d_rope=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_2d_rope = use_2d_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, curr_layer_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        # === 应用 2D RoPE 到 q/k ===
        if self.use_2d_rope and freqs_cis is not None:
            q, k = apply_rotary_emb_2d(q, k, freqs_cis=freqs_cis)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 应用 mask
        if curr_layer_mask is not None:
            mask = curr_layer_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ================= LayerScale =================
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ================= ViT Block with 2D RoPE =================
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., init_values=None, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_2d_rope=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop, use_2d_rope=use_2d_rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = act_layer()
        self.mlp2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, freqs_cis=None, curr_layer_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), freqs_cis, curr_layer_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp2(self.act(self.mlp(self.norm2(x))))))
        return x


# ================= LayerNorm 简化 =================
class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-5)


# ================= MAE-ViT with 2D RoPE =================
class MaskedAutoencoderViT(nn.Module):
    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 32],
                 patch_size=[8, 32],
                 embed_dim=768,
                 depth=4,
                 num_heads=6,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 use_2d_rope=False,
                 rope_mode='axial',  # 'axial' or 'mixed'
                 rope_theta=100.0):
        super().__init__()
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.use_2d_rope = use_2d_rope
        self.rope_mode = rope_mode

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # 使用支持 2D RoPE 的 Block
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer, use_2d_rope=use_2d_rope)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, nb_cls)
        self.unet = UNetAnySize(1, 1)

        # 初始化 2D RoPE 参数
        if use_2d_rope:
            if rope_mode == 'mixed':
                # Mixed RoPE: 可学习的频率参数
                freqs = init_random_2d_freqs(
                    dim=embed_dim // num_heads, 
                    num_heads=num_heads, 
                    theta=rope_theta
                )
                self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
                t_x, t_y = init_t_xy(end_x=self.grid_size[1], end_y=self.grid_size[0])
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
            else:
                # Axial RoPE: 固定频率
                freqs_cis = compute_axial_cis(
                    dim=embed_dim // num_heads,
                    end_x=self.grid_size[1],
                    end_y=self.grid_size[0],
                    theta=rope_theta
                )
                self.register_buffer('freqs_cis', freqs_cis)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_ratio=0.0, foreground_mask=None, use_masking=False):
        x = self.unet(x)
        x = self.layer_norm(x)
        x = self.patch_embed(x)

        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        # x = x + self.pos_embed

        # 计算或获取 2D RoPE 频率
        freqs_cis = None
        if self.use_2d_rope:
            if self.rope_mode == 'mixed':
                # 动态计算 Mixed RoPE
                if self.freqs_t_x.shape[0] != x.shape[1]:
                    t_x, t_y = init_t_xy(end_x=h, end_y=w)
                    t_x, t_y = t_x.to(x.device), t_y.to(x.device)
                else:
                    t_x, t_y = self.freqs_t_x, self.freqs_t_y
                freqs_cis = compute_mixed_cis(self.freqs, t_x, t_y, self.num_heads)
            else:
                # Axial RoPE
                if self.freqs_cis.shape[0] != x.shape[1]:
                    freqs_cis = compute_axial_cis(
                        dim=self.embed_dim // self.num_heads,
                        end_x=h,
                        end_y=w,
                        theta=100.0
                    )
                    freqs_cis = freqs_cis.to(x.device)
                else:
                    freqs_cis = self.freqs_cis

        # 前向传播所有 block
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis, curr_layer_mask=foreground_mask)

        x = self.norm(x)
        x = self.head(x)
        x = self.layer_norm(x)
        return x


# ================= 创建模型 =================
def create_model(nb_cls, img_size, use_2d_rope=True, rope_mode='axial', rope_theta=100.0, **kwargs):
    """
    创建 MAE-ViT 模型
    
    Args:
        nb_cls: 类别数
        img_size: 图像尺寸 [H, W]
        use_2d_rope: 是否使用 2D RoPE
        rope_mode: RoPE 模式，'axial' 或 'mixed'
        rope_theta: RoPE theta 参数
    """
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 use_2d_rope=use_2d_rope,
                                 rope_mode=rope_mode,
                                 rope_theta=rope_theta,
                                 **kwargs)
    return model


# ================= 测试 =================
if __name__ == "__main__":
    print("=" * 70)
    print("测试 1: 原始模型 (不使用 2D RoPE)")
    print("=" * 70)
    model_orig = create_model(80, [512, 32], use_2d_rope=False)
    x = torch.randn(2, 1, 512, 32)
    foreground_mask = torch.ones(2, model_orig.num_patches).bool()
    y = model_orig(x, foreground_mask=foreground_mask)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model_orig.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 70)
    print("测试 2: Axial RoPE 模型")
    print("=" * 70)
    model_axial = create_model(80, [512, 32], use_2d_rope=True, rope_mode='axial', rope_theta=100.0)
    y = model_axial(x, foreground_mask=foreground_mask)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model_axial.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 70)
    print("测试 3: Mixed RoPE 模型")
    print("=" * 70)
    model_mixed = create_model(80, [512, 32], use_2d_rope=True, rope_mode='mixed', rope_theta=10.0)
    y = model_mixed(x, foreground_mask=foreground_mask)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model_mixed.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 70)
    print("测试 4: 动态尺寸测试 (Axial RoPE)")
    print("=" * 70)
    x_large = torch.randn(1, 1, 1024, 64)
    y_large = model_axial(x_large)
    print(f"输入 shape: {x_large.shape}")
    print(f"输出 shape: {y_large.shape}")
    
    print("\n✅ 所有测试通过!")