"""课时 29-30：手算交叉熵。"""

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    # 4 个候选 token 的 logits。
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    target = torch.tensor([1])  # 正确 token 是索引 1。

    prob = torch.softmax(logits, dim=-1)
    loss = F.cross_entropy(logits, target)

    print("prob:", prob)
    print("cross entropy:", loss)
    print("manual:", -torch.log(prob[0, target.item()]))


if __name__ == "__main__":
    main()
