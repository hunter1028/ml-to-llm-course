"""课时 21-22：GRU 做序列分类。"""

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 PyTorch：python -m pip install torch") from exc


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)


def main() -> None:
    torch.manual_seed(42)

    num_samples = 1000
    seq_len = 20
    feature_dim = 4

    X = torch.randn(num_samples, seq_len, feature_dim)
    y = (X[:, -5:, :].mean(dim=(1, 2)) > X[:, :5, :].mean(dim=(1, 2))).long()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GRUClassifier(input_dim=feature_dim, hidden_dim=32, num_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(xb)
        print(epoch + 1, total_loss / len(dataset))


if __name__ == "__main__":
    main()
