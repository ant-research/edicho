import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from attnctrl.attn_utils import AttentionBase


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("Attention Ctrl at denoising steps: ", self.step_idx)
        print("Attention Ctrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)
        return out


def replace_tokens_by_corr(A, B, corr, replace_token_rate=1.0):
    batch_size, num_tokens, token_dim = B.shape
    replace_token_count = int(num_tokens * replace_token_rate)
    replace_indices_source = torch.randperm(num_tokens)[:replace_token_count]
    replace_indices_target = corr[replace_indices_source]
    temp = A.clone().to(B.device)
    for i in range(batch_size):
        B[i, replace_indices_target] = temp[i, replace_indices_source]
    return B


def replace_tokens_by_corr_and_mask(X, Y, corr, mask_s=None, mask_t=None):
    B, N, C = Y.shape
    masks_expand = mask_s.unsqueeze(0).expand(B, -1)
    maskt_expand = mask_t.unsqueeze(0).expand(B, -1)
    X_masked = X[masks_expand].view(B, -1, C)
    Y_masked = Y[maskt_expand].view(B, -1, C)

    for i in range(B):
        Y_masked[i] = X_masked[i, corr]
        Y[i, mask_t] = Y_masked[i]
    return Y


class EdichoSelfAttentionControl(MutualSelfAttentionControl):
    def __init__(self, start_step=4, end_step=50, start_layer=10,
                 layer_idx=None, step_idx=None, total_steps=50,
                 mask_s=None, mask_t=None, model_type="SD",
                 mask_query_coords=None, mask_value_coords=None,
                 edit_strength=0.8, replace_token_rate=0.0,
                 dift_feats=None, hash_key=None,
                 dift_corr_extractor=None,
                 imgW=None, imgH=None,
                 ):
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        MODEL_TYPE2LAYER = {
            "SD": 16,
            "SDXL": 70
        }
        self.end_step = end_step
        self.step_idx = list(range(self.start_step, self.end_step))

        self.edit_strength = edit_strength
        self.replace_token_rate = replace_token_rate
        self.total_layers = MODEL_TYPE2LAYER[model_type]
        self.mask_s = mask_s
        self.mask_t = mask_t
        self.mask_query_coords = mask_query_coords
        self.mask_value_coords = mask_value_coords

        self.dift_feats = dift_feats
        self.hash_key = hash_key
        self.dift_corr_extractor = dift_corr_extractor
        self.corrs = []
        self.imgW = imgW
        self.imgH = imgH

    def self_attention_qkv(self, q, k, v, num_heads, no_bg=True, use_mask=False, **kwargs):
        ratio = int(np.sqrt(self.imgH * self.imgW // q.shape[1]))
        H = self.imgH // ratio
        W = self.imgW // ratio

        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if use_mask:
            assert self.mask_s is not None and self.mask_t is not None
            mask_s = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask_s = F.interpolate(mask_s, (H, W)).flatten(0).unsqueeze(0)
            mask_s = mask_s.flatten().to(sim)
            sim_fg = sim + mask_s.masked_fill(mask_s == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask_s.masked_fill(mask_s == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)

        if use_mask:
            out_fg, out_bg = out.chunk(2, 0)
            mask_t = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W)).to(out)
            mask_t = mask_t.reshape(-1, 1)
            out = out_fg * mask_t + out_bg * (1 - mask_t)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if is_cross or int(self.cur_step) not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        ratio = int(np.sqrt(self.imgH * self.imgW // q.shape[1]))
        H = self.imgH // ratio
        W = self.imgW // ratio

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], num_heads, **kwargs)

        corr = self.dift_corr_extractor.extract_corr_from_dift(self.dift_feats, self.hash_key,
                                                               token_H=H, token_W=W, return2D=False)

        def edicho_attnctrl(q, k, v, num_heads, corr, replace_token_rate, **kwargs):
            q_source = rearrange(q[:num_heads], '(b h) n d -> b n (h d)', h=num_heads)
            q_target = rearrange(q[num_heads:], '(b h) n d -> b n (h d)', h=num_heads)
            q_warped_source = replace_tokens_by_corr(q_source, q_target, corr, replace_token_rate)
            q_warped_source = rearrange(q_warped_source, 'b n (h d) -> (b h) n d', h=num_heads)
            out_target = self.self_attention_qkv(q_warped_source, k[:num_heads], v[:num_heads], num_heads, **kwargs)
            return out_target

        out_u_target = edicho_attnctrl(qu, ku, vu, num_heads, corr, self.replace_token_rate, **kwargs)
        out_c_target = edicho_attnctrl(qc, kc, vc, num_heads, corr, self.replace_token_rate, **kwargs)
        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)

        return out
