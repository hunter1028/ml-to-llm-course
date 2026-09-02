"""课时 23-24：MLP 与 CNN 处理图片输入的差异。"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )

    x = torch.randn(8, 1, 28, 28)
    print("MLP output:", mlp(x).shape)
    print("CNN output:", cnn(x).shape)
    print("MLP parameters:", sum(p.numel() for p in mlp.parameters()))
    print("CNN parameters:", sum(p.numel() for p in cnn.parameters()))


if __name__ == "__main__":
    main()
