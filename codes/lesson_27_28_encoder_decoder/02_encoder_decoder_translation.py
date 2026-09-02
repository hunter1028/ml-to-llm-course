"""课时 27-28：Encoder-Decoder 做文本到文本生成。"""

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 transformers：python -m pip install transformers") from exc


def main() -> None:
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    prompt = "Translate to German: The model reads the input first."
    batch = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**batch, max_new_tokens=40)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
