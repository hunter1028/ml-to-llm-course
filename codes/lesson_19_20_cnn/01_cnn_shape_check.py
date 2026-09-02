"""课时 19-20：逐层检查 CNN 特征图形状。"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )


def main() -> None:
    model = SmallCNN()
    x = torch.randn(8, 1, 28, 28)
    print("input", x.shape)
    for layer in model.features:
        x = layer(x)
        print(layer.__class__.__name__, x.shape)


if __name__ == "__main__":
    main()
