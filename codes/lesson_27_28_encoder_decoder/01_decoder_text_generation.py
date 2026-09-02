"""课时 27-28：Decoder 做文本生成。"""

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 transformers：python -m pip install transformers") from exc


def main() -> None:
    generator = pipeline(
        task="text-generation",
        model="openai-community/gpt2",
    )

    result = generator(
        "Machine learning systems should be evaluated by",
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    print(result[0]["generated_text"])


if __name__ == "__main__":
    main()
