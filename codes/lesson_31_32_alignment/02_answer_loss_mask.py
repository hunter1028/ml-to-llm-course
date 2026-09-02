"""课时 31-32：简化的回答区域 Loss Mask。"""

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    # logits: [B, N, V]
    logits = torch.randn(2, 6, 100)
    labels = torch.tensor([
        [-100, -100, -100, 12, 33, 9],
        [-100, -100, 18, 25, 7, 2],
    ])

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    print(loss)


if __name__ == "__main__":
    main()
