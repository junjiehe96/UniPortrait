# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.lora import LoRALinearLayer


class AttentionArgs(object):
    def __init__(self) -> None:
        # ip condition
        self.ip_scale = 0.0
        self.ip_mask = None  # ip attention mask

        # faceid condition
        self.lora_scale = 0.0  # lora for single faceid
        self.multi_id_lora_scale = 0.0  # lora for multiple faceids
        self.faceid_scale = 0.0
        self.num_faceids = 0
        self.faceid_mask = None  # faceid attention mask; if not None, it will override the routing map

        # style aligned
        self.enable_share_attn: bool = False
        self.adain_queries_and_keys: bool = False
        self.shared_score_scale: float = 1.0
        self.shared_score_shift: float = 0.0

    def reset(self):
        # ip condition
        self.ip_scale = 0.0
        self.ip_mask = None  # ip attention mask

        # faceid condition
        self.lora_scale = 0.0  # lora for single faceid
        self.multi_id_lora_scale = 0.0  # lora for multiple faceids
        self.faceid_scale = 0.0
        self.num_faceids = 0
        self.faceid_mask = None  # faceid attention mask; if not None, it will override the routing map

        # style aligned
        self.enable_share_attn: bool = False
        self.adain_queries_and_keys: bool = False
        self.shared_score_scale: float = 1.0
        self.shared_score_shift: float = 0.0

    def __repr__(self):
        indent_str = '    '
        s = f",\n{indent_str}".join(f"{attr}={value}" for attr, value in vars(self).items())
        return self.__class__.__name__ + '(' + f'\n{indent_str}' + s + ')'


attn_args = AttentionArgs()


def expand_first(feat, scale=1., ):
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat, dim=2, scale=1.):
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat):
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


class UniPortraitLoRAAttnProcessor2_0(nn.Module):

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
            rank=128,
            network_alpha=None,
    ):
        super().__init__()

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        self.to_q_multi_id_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_multi_id_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_multi_id_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_multi_id_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if attn_args.lora_scale > 0.0:
            query = query + attn_args.lora_scale * self.to_q_lora(hidden_states)
            key = key + attn_args.lora_scale * self.to_k_lora(encoder_hidden_states)
            value = value + attn_args.lora_scale * self.to_v_lora(encoder_hidden_states)
        elif attn_args.multi_id_lora_scale > 0.0:
            query = query + attn_args.multi_id_lora_scale * self.to_q_multi_id_lora(hidden_states)
            key = key + attn_args.multi_id_lora_scale * self.to_k_multi_id_lora(encoder_hidden_states)
            value = value + attn_args.multi_id_lora_scale * self.to_v_multi_id_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn_args.enable_share_attn:
            if attn_args.adain_queries_and_keys:
                query = adain(query)
                key = adain(key)
            key = concat_first(key, -2, scale=attn_args.shared_score_scale)
            value = concat_first(value, -2)
            if attn_args.shared_score_shift != 0:
                attention_mask = torch.zeros_like(key[:, :, :, :1]).transpose(-1, -2)  # b, h, 1, k
                attention_mask[:, :, :, query.shape[2]:] += attn_args.shared_score_shift
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
                )
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
                )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        output_hidden_states = attn.to_out[0](hidden_states)
        if attn_args.lora_scale > 0.0:
            output_hidden_states = output_hidden_states + attn_args.lora_scale * self.to_out_lora(hidden_states)
        elif attn_args.multi_id_lora_scale > 0.0:
            output_hidden_states = output_hidden_states + attn_args.multi_id_lora_scale * self.to_out_multi_id_lora(
                hidden_states)
        hidden_states = output_hidden_states

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class UniPortraitLoRAIPAttnProcessor2_0(nn.Module):

    def __init__(self, hidden_size, cross_attention_dim=None, rank=128, network_alpha=None,
                 num_ip_tokens=4, num_faceid_tokens=16):
        super().__init__()

        self.num_ip_tokens = num_ip_tokens
        self.num_faceid_tokens = num_faceid_tokens

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.to_k_faceid = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_faceid = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.to_q_multi_id_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_multi_id_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_multi_id_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_multi_id_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        self.to_q_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        )
        self.to_k_router = nn.Sequential(
            nn.Linear(cross_attention_dim or hidden_size, (cross_attention_dim or hidden_size) * 2),
            nn.GELU(),
            nn.Linear((cross_attention_dim or hidden_size) * 2, hidden_size, bias=False),
        )
        self.aggr_router = nn.Linear(num_faceid_tokens, 1)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # split hidden states
            faceid_end = encoder_hidden_states.shape[1]
            ip_end = faceid_end - self.num_faceid_tokens * attn_args.num_faceids
            text_end = ip_end - self.num_ip_tokens

            prompt_hidden_states = encoder_hidden_states[:, :text_end]
            ip_hidden_states = encoder_hidden_states[:, text_end: ip_end]
            faceid_hidden_states = encoder_hidden_states[:, ip_end: faceid_end]

            encoder_hidden_states = prompt_hidden_states
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # for router
        if attn_args.num_faceids > 1:
            router_query = self.to_q_router(hidden_states)  # bs, s*s, dim
            router_hidden_states = faceid_hidden_states.reshape(batch_size, attn_args.num_faceids,
                                                                self.num_faceid_tokens, -1)  # bs, num, id_tokens, d
            router_hidden_states = self.aggr_router(router_hidden_states.transpose(-1, -2)).squeeze(-1)  # bs, num, d
            router_key = self.to_k_router(router_hidden_states)  # bs, num, dim
            router_logits = torch.bmm(router_query, router_key.transpose(-1, -2))  # bs, s*s, num
            index = router_logits.max(dim=-1, keepdim=True)[1]
            routing_map = torch.zeros_like(router_logits).scatter_(-1, index, 1.0)
            routing_map = routing_map.transpose(1, 2).unsqueeze(-1)  # bs, num, s*s, 1
        else:
            routing_map = hidden_states.new_ones(size=(1, 1, hidden_states.shape[1], 1))

        # for text
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if attn_args.lora_scale > 0.0:
            query = query + attn_args.lora_scale * self.to_q_lora(hidden_states)
            key = key + attn_args.lora_scale * self.to_k_lora(encoder_hidden_states)
            value = value + attn_args.lora_scale * self.to_v_lora(encoder_hidden_states)
        elif attn_args.multi_id_lora_scale > 0.0:
            query = query + attn_args.multi_id_lora_scale * self.to_q_multi_id_lora(hidden_states)
            key = key + attn_args.multi_id_lora_scale * self.to_k_multi_id_lora(encoder_hidden_states)
            value = value + attn_args.multi_id_lora_scale * self.to_v_multi_id_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        if attn_args.ip_scale > 0.0:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=attn.scale
            )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            if attn_args.ip_mask is not None:
                ip_mask = attn_args.ip_mask
                h, w = ip_mask.shape[-2:]
                ratio = (h * w / query.shape[2]) ** 0.5
                ip_mask = torch.nn.functional.interpolate(ip_mask, scale_factor=1 / ratio,
                                                          mode='nearest').reshape(
                    [1, -1, 1])
                ip_hidden_states = ip_hidden_states * ip_mask

            if attn_args.enable_share_attn:
                ip_hidden_states[0] = 0.
                ip_hidden_states[batch_size // 2] = 0.
        else:
            ip_hidden_states = torch.zeros_like(hidden_states)

        # for faceid-adapter
        if attn_args.faceid_scale > 0.0:
            faceid_key = self.to_k_faceid(faceid_hidden_states)
            faceid_value = self.to_v_faceid(faceid_hidden_states)

            faceid_query = query[:, None].expand(-1, attn_args.num_faceids, -1, -1,
                                                 -1)  # 2*bs, num, heads, s*s, dim/heads
            faceid_key = faceid_key.view(batch_size, attn_args.num_faceids, self.num_faceid_tokens, attn.heads,
                                         head_dim).transpose(2, 3)
            faceid_value = faceid_value.view(batch_size, attn_args.num_faceids, self.num_faceid_tokens, attn.heads,
                                             head_dim).transpose(2, 3)

            faceid_hidden_states = F.scaled_dot_product_attention(
                faceid_query, faceid_key, faceid_value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=attn.scale
            )  # 2*bs, num, heads, s*s, dim/heads

            faceid_hidden_states = faceid_hidden_states.transpose(2, 3).reshape(batch_size, attn_args.num_faceids, -1,
                                                                                attn.heads * head_dim)
            faceid_hidden_states = faceid_hidden_states.to(query.dtype)  # 2*bs, num, s*s, dim

            if attn_args.faceid_mask is not None:
                faceid_mask = attn_args.faceid_mask  # 1, num, h, w
                h, w = faceid_mask.shape[-2:]
                ratio = (h * w / query.shape[2]) ** 0.5
                faceid_mask = F.interpolate(faceid_mask, scale_factor=1 / ratio,
                                            mode='bilinear').flatten(2).unsqueeze(-1)  # 1, num, s*s, 1
                faceid_mask = faceid_mask / faceid_mask.sum(1, keepdim=True).clip(min=1e-3)  # 1, num, s*s, 1
                faceid_hidden_states = (faceid_mask * faceid_hidden_states).sum(1)  # 2*bs, s*s, dim
            else:
                faceid_hidden_states = (routing_map * faceid_hidden_states).sum(1)  # 2*bs, s*s, dim

            if attn_args.enable_share_attn:
                faceid_hidden_states[0] = 0.
                faceid_hidden_states[batch_size // 2] = 0.
        else:
            faceid_hidden_states = torch.zeros_like(hidden_states)

        hidden_states = hidden_states + \
                        attn_args.ip_scale * ip_hidden_states + \
                        attn_args.faceid_scale * faceid_hidden_states

        # linear proj
        output_hidden_states = attn.to_out[0](hidden_states)
        if attn_args.lora_scale > 0.0:
            output_hidden_states = output_hidden_states + attn_args.lora_scale * self.to_out_lora(hidden_states)
        elif attn_args.multi_id_lora_scale > 0.0:
            output_hidden_states = output_hidden_states + attn_args.multi_id_lora_scale * self.to_out_multi_id_lora(
                hidden_states)
        hidden_states = output_hidden_states

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# for controlnet
class UniPortraitCNAttnProcessor2_0:
    def __init__(self, num_ip_tokens=4, num_faceid_tokens=16):

        self.num_ip_tokens = num_ip_tokens
        self.num_faceid_tokens = num_faceid_tokens

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            text_end = encoder_hidden_states.shape[1] - self.num_faceid_tokens * attn_args.num_faceids \
                       - self.num_ip_tokens
            encoder_hidden_states = encoder_hidden_states[:, :text_end]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
