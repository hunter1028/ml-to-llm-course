"""课时 25-26：手写缩放点积注意力。"""

import math

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def scaled_dot_product_attention(q, k, v, mask=None):
    # q: [B, H, Nq, Dh]
    # k: [B, H, Nk, Dh]
    # v: [B, H, Nk, Dv]
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        # mask=True 表示允许关注；False 表示遮住。
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


def main() -> None:
    B, H, N, Dh = 2, 4, 6, 8
    q = torch.randn(B, H, N, Dh)
    k = torch.randn(B, H, N, Dh)
    v = torch.randn(B, H, N, Dh)

    # 下三角因果掩码：[1, 1, N, N]，可广播到 batch 和 head。
    causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool))[None, None]
    out, attn = scaled_dot_product_attention(q, k, v, causal_mask)

    print(out.shape)   # [2, 4, 6, 8]
    print(attn.shape)  # [2, 4, 6, 6]


if __name__ == "__main__":
    main()
