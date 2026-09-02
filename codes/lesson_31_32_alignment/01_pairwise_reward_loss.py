"""课时 31-32：奖励模型成对 Loss。"""

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    reward_chosen = torch.tensor([2.0, 0.5, 1.2])
    reward_rejected = torch.tensor([0.5, 0.8, -0.1])

    # -log(sigmoid(chosen - rejected)) 的稳定实现。
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    accuracy = (reward_chosen > reward_rejected).float().mean()

    print("pairwise loss:", loss.item())
    print("ranking accuracy:", accuracy.item())


if __name__ == "__main__":
    main()
