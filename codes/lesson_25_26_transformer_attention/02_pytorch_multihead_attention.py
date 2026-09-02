"""课时 25-26：使用 PyTorch 多头注意力。"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    mha = nn.MultiheadAttention(
        embed_dim=64,
        num_heads=4,
        batch_first=True,
    )

    x = torch.randn(2, 10, 64)  # [B, N, D]
    causal_mask = torch.triu(
        torch.ones(10, 10, dtype=torch.bool),
        diagonal=1,
    )

    out, weights = mha(
        query=x,
        key=x,
        value=x,
        attn_mask=causal_mask,
        need_weights=True,
    )

    print(out.shape)      # [2, 10, 64]
    print(weights.shape)  # 默认对 head 求平均后为 [2, 10, 10]


if __name__ == "__main__":
    main()
