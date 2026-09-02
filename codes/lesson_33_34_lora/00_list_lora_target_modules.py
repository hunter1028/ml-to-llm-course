"""课时 33-34：查看模型中可作为 LoRA target_modules 的线性层。"""

try:
    from transformers import AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 transformers：python -m pip install transformers") from exc


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    for name, module in model.named_modules():
        if "proj" in name:
            print(name, type(module).__name__)


if __name__ == "__main__":
    main()
