# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
import torch.nn as nn


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, attention_mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latents (torch.Tensor): latent features
                shape (b, n2, D)
            attention_mask (torch.Tensor): attention mask
                shape (b, n1, 1)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        if attention_mask is not None:
            attention_mask = attention_mask.transpose(1, 2)  # (b, 1, n1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :, :1]).repeat(1, 1, l)],
                                       dim=2)  # b, 1, n1+n2
            attention_mask = (attention_mask - 1.) * 100.  # 0 means kept and -100 means dropped
            attention_mask = attention_mask.unsqueeze(1)
            weight = weight + attention_mask  # b, h, n2, n1+n2

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class UniPortraitFaceIDResampler(torch.nn.Module):
    def __init__(
            self,
            intrinsic_id_embedding_dim=512,
            structure_embedding_dim=64 + 128 + 256 + 1280,
            num_tokens=16,
            depth=6,
            dim=768,
            dim_head=64,
            heads=12,
            ff_mult=4,
            output_dim=768,
    ):
        super().__init__()

        self.latents = torch.nn.Parameter(torch.randn(1, num_tokens, dim) / dim ** 0.5)

        self.proj_id = torch.nn.Sequential(
            torch.nn.Linear(intrinsic_id_embedding_dim, intrinsic_id_embedding_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(intrinsic_id_embedding_dim * 2, dim),
        )
        self.proj_clip = torch.nn.Sequential(
            torch.nn.Linear(structure_embedding_dim, structure_embedding_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(structure_embedding_dim * 2, dim),
        )

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)

    def forward(
            self,
            intrinsic_id_embeds,
            structure_embeds,
            structure_scale=1.0,
            intrinsic_id_attention_mask=None,
            structure_attention_mask=None
    ):

        latents = self.latents.repeat(intrinsic_id_embeds.size(0), 1, 1)

        intrinsic_id_embeds = self.proj_id(intrinsic_id_embeds)
        structure_embeds = self.proj_clip(structure_embeds)

        for attn1, attn2, ff in self.layers:
            latents = attn1(intrinsic_id_embeds, latents, intrinsic_id_attention_mask) + latents
            latents = structure_scale * attn2(structure_embeds, latents, structure_attention_mask) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
