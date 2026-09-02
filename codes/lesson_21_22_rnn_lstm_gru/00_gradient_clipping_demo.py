"""课时 21-22：梯度裁剪最小示例。"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    model = nn.GRU(input_size=4, hidden_size=8, batch_first=True)
    x = torch.randn(16, 20, 4)
    out, _ = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print("global grad norm before clipping:", float(before))
    print("clip_grad_norm_ 已把全局梯度范数限制到 1.0 附近")


if __name__ == "__main__":
    main()
