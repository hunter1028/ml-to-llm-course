"""课时 33-34：准备售后工单 SFT 数据集。"""

try:
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 datasets：python -m pip install datasets") from exc


samples = [
    {
        "messages": [
            {"role": "system", "content": "你是售后工单助手，只输出合法 JSON。"},
            {"role": "user", "content": "耳机左耳无声，订单号 A1024，希望换货。"},
            {
                "role": "assistant",
                "content": '{"intent":"质量问题","urgency":"中","order_id":"A1024","action":"创建换货工单"}',
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是售后工单助手，只输出合法 JSON。"},
            {"role": "user", "content": "包裹显示签收但没收到，订单号 B7788。"},
            {
                "role": "assistant",
                "content": '{"intent":"物流异常","urgency":"高","order_id":"B7788","action":"创建物流核查工单"}',
            },
        ]
    },
]


def build_dataset():
    dataset = Dataset.from_list(samples)
    split = dataset.train_test_split(test_size=0.5, seed=42)
    return split["train"], split["test"]


def main() -> None:
    train_dataset, eval_dataset = build_dataset()
    print(train_dataset)
    print(eval_dataset)


if __name__ == "__main__":
    main()
