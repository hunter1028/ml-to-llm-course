# 课时 29-30：大模型预训练逻辑代码

| 文件 | 对应内容 | 运行方式 |
|---|---|---|
| `00_cross_entropy_by_hand.py` | 手算交叉熵 | `python lesson_29_30_pretraining/00_cross_entropy_by_hand.py` |
| `01_causal_lm_label_shift.py` | 因果语言模型标签右移 | `python lesson_29_30_pretraining/01_causal_lm_label_shift.py` |
| `02_context_pooling_embeddings.py` | 提取上下文表示并做 pooling | `python lesson_29_30_pretraining/02_context_pooling_embeddings.py` |
| `03_hallucination_test_cases.py` | 幻觉测试用例表 | `python lesson_29_30_pretraining/03_hallucination_test_cases.py` |

`02_context_pooling_embeddings.py` 需要 PyTorch 和 Transformers，首次运行会下载模型。
