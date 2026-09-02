"""课时 33-34：合并 LoRA 权重。"""

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 transformers 和 peft：python -m pip install transformers peft") from exc


BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "outputs/ticket-lora/final-adapter"
MERGED_DIR = "outputs/ticket-lora/merged-model"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(MERGED_DIR)


if __name__ == "__main__":
    main()
