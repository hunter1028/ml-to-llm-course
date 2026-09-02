"""课时 25-26：实现正弦位置编码。"""

import math

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


def sinusoidal_position_encoding(max_len: int, d_model: int):
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def main() -> None:
    pe = sinusoidal_position_encoding(max_len=128, d_model=64)
    print(pe.shape)  # [128, 64]
    print(pe[:3, :8].round(decimals=4))


if __name__ == "__main__":
    main()
