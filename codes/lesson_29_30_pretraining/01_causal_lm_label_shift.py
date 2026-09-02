"""课时 29-30：构造因果语言模型标签。"""

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def main() -> None:
    input_ids = torch.tensor([
        [101, 2057, 2293, 3322, 102],
        [101, 1921, 3698, 1962, 102],
    ])

    # 直观演示右移关系。
    model_input = input_ids[:, :-1]
    target = input_ids[:, 1:]

    print("model_input:")
    print(model_input)
    print("target:")
    print(target)
    print("model_input.shape:", model_input.shape)
    print("target.shape:", target.shape)


if __name__ == "__main__":
    main()
