"""课时 27-28：观察 Encoder 模型类和输出形状。"""

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 torch 和 transformers：python -m pip install torch transformers") from exc


def main() -> None:
    model_id = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    batch = tokenizer(
        ["Transformers build contextual representations."],
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        output = model(**batch)

    print(batch["input_ids"].shape)
    print(output.last_hidden_state.shape)
    print("[batch_size, sequence_length, hidden_size]")


if __name__ == "__main__":
    main()
