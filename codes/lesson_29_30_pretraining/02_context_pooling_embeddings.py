"""课时 29-30：提取上下文表示并做简单 pooling。"""

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 torch 和 transformers：python -m pip install torch transformers") from exc


def main() -> None:
    model_id = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    texts = [
        "A model learns patterns from data.",
        "Machine learning discovers patterns in datasets.",
    ]

    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        hidden = model(**batch).last_hidden_state

    mask = batch["attention_mask"].unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    emb = summed / count
    emb = F.normalize(emb, p=2, dim=-1)

    similarity = emb @ emb.T
    print(emb.shape)
    print(similarity)


if __name__ == "__main__":
    main()
