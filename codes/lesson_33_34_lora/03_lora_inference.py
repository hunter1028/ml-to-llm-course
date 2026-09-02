"""课时 33-34：加载 LoRA 适配器并推理。"""

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 transformers 和 peft：python -m pip install transformers peft") from exc


BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "outputs/ticket-lora/final-adapter"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    messages = [
        {"role": "system", "content": "你是售后工单助手，只输出合法 JSON。"},
        {"role": "user", "content": "充电器发热并有焦味，订单号 C9001。"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer_ids = outputs[0, inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(answer_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
