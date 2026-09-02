"""课时 33-34：LoRA 微调售后工单 JSON 抽取模型。"""

try:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "请先安装训练依赖：python -m pip install torch transformers datasets peft trl accelerate"
    ) from exc


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


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


def main() -> None:
    dataset = Dataset.from_list(samples)
    split = dataset.train_test_split(test_size=0.5, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir="outputs/ticket-lora",
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        max_length=512,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    model.print_trainable_parameters()
    trainer.save_model("outputs/ticket-lora/final-adapter")
    tokenizer.save_pretrained("outputs/ticket-lora/final-adapter")


if __name__ == "__main__":
    main()
