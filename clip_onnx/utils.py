import torch.nn.functional as F
import torch
from torch import nn


class Textual(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # needs .float() before .argmax(  ) to work
        x = x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection

        return x


def attention(self, x: torch.Tensor):
    # onnx doesn't like multi_head_attention_forward so this is a reimplementation
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    q, k, v = (torch.einsum("tbh, oh -> tbo", x, self.attn.in_proj_weight) + self.attn.in_proj_bias).contiguous().chunk(
        3, dim=-1)
    tgt_len = q.shape[0]
    bsz = q.shape[1]
    num_heads = self.attn.num_heads
    head_dim = q.shape[2] // num_heads
    attn_output = scaled_dot_product_attention(
        q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1),
        k.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1),
        v.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1), self.attn_mask, 0.0
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(q.shape)
    attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)
    return attn_output

def scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p):
    if attn_mask is None:
        attn_weight = torch.softmax(Q @ K.transpose(-2, -1) / Q.size(-1)**0.5, dim=-1)
    else:
        attn_weight = torch.softmax(Q @ K.transpose(-2, -1) / Q.size(-1)**0.5 + attn_mask[None, ...], dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p) # this is always 0.0 in CLIP so I comment it out.
    return attn_weight @ V


DEFAULT_EXPORT = dict(input_names=['input'], output_names=['output'],
                      export_params=True, verbose=False, opset_version=12,
                      do_constant_folding=True,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
