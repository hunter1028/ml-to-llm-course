# 课时 33-34：LoRA 轻量化微调代码

| 文件 | 对应内容 | 运行方式 |
|---|---|---|
| `00_list_lora_target_modules.py` | 查看 LoRA 可挂载模块 | `python lesson_33_34_lora/00_list_lora_target_modules.py` |
| `01_prepare_ticket_dataset.py` | 准备售后工单 SFT 数据集 | `python lesson_33_34_lora/01_prepare_ticket_dataset.py` |
| `02_train_lora_ticket_adapter.py` | LoRA 微调完整训练入口 | `python lesson_33_34_lora/02_train_lora_ticket_adapter.py` |
| `03_lora_inference.py` | 加载适配器推理 | `python lesson_33_34_lora/03_lora_inference.py` |
| `04_merge_lora_adapter.py` | 合并 LoRA 权重 | `python lesson_33_34_lora/04_merge_lora_adapter.py` |
| `05_json_evaluator.py` | JSON 输出评测器 | `python lesson_33_34_lora/05_json_evaluator.py` |

训练和推理脚本需要 PyTorch、Transformers、Datasets、PEFT、TRL、Accelerate。首次运行会下载模型。
