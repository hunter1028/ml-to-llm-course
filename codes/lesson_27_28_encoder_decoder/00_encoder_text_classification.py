"""课时 27-28：Encoder 做文本分类。"""

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 transformers：python -m pip install transformers") from exc


def main() -> None:
    classifier = pipeline(
        task="text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )

    result = classifier("The service was fast and reliable.")
    print(result)


if __name__ == "__main__":
    main()
