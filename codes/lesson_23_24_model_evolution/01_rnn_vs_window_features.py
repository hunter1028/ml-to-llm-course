"""课时 23-24：窗口统计特征与 RNN 序列输入对比。"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    batch_size = 8
    seq_len = 20
    feature_dim = 4
    seq = torch.randn(batch_size, seq_len, feature_dim)

    features = torch.cat([
        seq.mean(dim=1),
        seq.max(dim=1).values,
        seq[:, -1, :],
    ], dim=1)

    rnn = nn.GRU(input_size=feature_dim, hidden_size=64, batch_first=True)
    out, h_n = rnn(seq)
    last_hidden = h_n[-1]

    print("window features shape:", features.shape)
    print("rnn out shape:", out.shape)
    print("last hidden shape:", last_hidden.shape)


if __name__ == "__main__":
    main()
