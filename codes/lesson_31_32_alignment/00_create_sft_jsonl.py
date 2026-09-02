"""课时 31-32：构造一条 SFT JSONL 数据。"""

import json
from pathlib import Path


samples = [
    {
        "messages": [
            {"role": "system", "content": "你是工单提取助手。"},
            {"role": "user", "content": "昨天重复扣款，希望退款。"},
            {
                "role": "assistant",
                "content": "问题：重复扣款\n时间：昨天\n诉求：退款",
            },
        ]
    }
]


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs" / "sft_demo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
