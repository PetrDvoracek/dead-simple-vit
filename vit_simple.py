import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        print(self.proj)

    def forward(self, x):
        # x: (batch, 3, H, W) -> (batch, embed_dim, H/P, W/P)
        print(x.shape)
        x = self.proj(x)

        # Flatten spatial dims: (batch, embed_dim, N) -> (batch, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head self-attention (simplified to single head)."""

    def __init__(self, dim):
        super().__init__()
        # self.qkv = nn.Linear(dim, dim * 3)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x):
        B, N, C = x.shape

        q, k, v = self.q(x), self.k(x), self.v(x)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, v)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(self, patch_size=16, embed_dim=512, num_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks = nn.Sequential(TransformerBlock(embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        # Prepend CLS token B x 1 x C on first token position
        cls = self.cls_token.expand(x.shape[0], 1, self.embed_dim)
        x = torch.cat([cls, x], dim=1)

        x = self.blocks(x)

        # Classify using CLS token, it is on first token position
        return self.head(x[:, 0])
