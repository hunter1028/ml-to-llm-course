# 课程示例代码

代码按课时分目录存放，避免后续脚本越来越多时堆在根目录。

## 目录结构

```text
codes/
├── lesson_00_intro/                         两个小 AI 例子
├── lesson_01_02_ai_intro/                   课时 01-02：行业认知与学习方法论
├── lesson_03_04_math_foundations/           课时 03-04：AI 核心数学基础
├── lesson_05_06_ml_basics/                  课时 05-06：机器学习基础范式与通用原理
├── lesson_07_08_linear_models/              课时 07-08：线性模型理论与实战
├── lesson_09_10_clustering_pca/             课时 09-10：聚类与降维
├── lesson_11_12_tree_models/                课时 11-12：树模型
├── lesson_13_14_gbdt_xgboost_lightgbm/      课时 13-14：GBDT / XGBoost / LightGBM
├── lesson_15_16_ml_engineering/             课时 15-16：模型工程与决策优化
├── lesson_17_18_deep_learning_mlp/          课时 17-18：深度学习与全连接网络
├── lesson_19_20_cnn/                        课时 19-20：CNN
├── lesson_21_22_rnn_lstm_gru/               课时 21-22：RNN / LSTM / GRU
├── lesson_23_24_model_evolution/            课时 23-24：CNN / RNN 对比复盘
├── lesson_25_26_transformer_attention/      课时 25-26：Transformer 与自注意力
├── lesson_27_28_encoder_decoder/            课时 27-28：Encoder / Decoder
├── lesson_29_30_pretraining/                课时 29-30：大模型预训练
├── lesson_31_32_alignment/                  课时 31-32：SFT / RM / RLHF
├── lesson_33_34_lora/                       课时 33-34：LoRA 微调
├── lesson_35_36_llm_production/             课时 35-36：大模型工业落地
├── lesson_37_38_capstone_project/           课时 37-38：综合项目
├── main.py                                  代码目录说明入口
├── pyproject.toml                           Python 依赖配置
└── uv.lock                                  uv 锁文件
```

## 新增代码放置规则

- 所有课程示例脚本统一放在 `codes/` 下，不再散落到课程 Markdown 同级目录。
- 每两个课时一个目录，命名格式为 `lesson_课时范围_主题英文名/`，例如 `lesson_09_10_clustering_pca/`。
- Markdown 中保留必要的短代码片段，同时在代码块前加“完整脚本见”链接，指向对应的 `codes/lesson_xx_xx_*/` 文件。
- 脚本运行产生的图片、模型、jsonl 等临时结果放到对应课时目录的 `outputs/`，该目录不进入 Git。

## 安装依赖

基础机器学习脚本使用 NumPy、Pandas、Matplotlib、Scikit-learn。推荐在 `codes/` 目录下使用 uv：

```bash
cd /Users/liuzhicheng/Documents/01pgm/ml-course/codes
uv sync
```

也可以使用 pip：

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

部分后续课时需要额外依赖：

```bash
# 13-14、37-38 中的增强树模型
python -m pip install xgboost lightgbm

# 17-32 中的深度学习与 Transformer 示例
python -m pip install torch transformers

# 33-34 中的 LoRA 微调示例
python -m pip install torch transformers datasets peft trl accelerate
```

## 两个小 AI 例子

```bash
python lesson_00_intro/01_ml_linear_regression.py
python lesson_00_intro/02_dl_mlp_digits.py
```

## 课时 01-02：行业认知与学习方法论

```bash
python lesson_01_02_ai_intro/00_house_price_linear_regression.py
```

## 课时 03-04：AI 核心数学基础

```bash
python lesson_03_04_math_foundations/00_vector_matrix_prediction.py
python lesson_03_04_math_foundations/01_batch_prediction.py
python lesson_03_04_math_foundations/02_cosine_similarity.py
python lesson_03_04_math_foundations/03_normal_distribution.py
python lesson_03_04_math_foundations/04_gradient_descent_demo.py
```

## 课时 05-06：机器学习基础范式与通用原理

```bash
python lesson_05_06_ml_basics/00_environment_check.py
python lesson_05_06_ml_basics/01_manual_train_val_test_split.py
python lesson_05_06_ml_basics/02_baseline_and_ridge.py
python lesson_05_06_ml_basics/03_manual_l2_loss.py
```

## 课时 07-08：线性模型理论与实战

```bash
python lesson_07_08_linear_models/00_normal_equation_demo.py
python lesson_07_08_linear_models/01_linear_regression_diabetes.py
python lesson_07_08_linear_models/02_logistic_regression_breast_cancer.py
python lesson_07_08_linear_models/03_l1_l2_huber_regression.py
```

## 课时 09-10：传统聚类与降维算法

```bash
python lesson_09_10_clustering_pca/00_kmeans_pca_user_segments.py
python lesson_09_10_clustering_pca/01_dbscan_outliers.py
```

## 课时 11-12：树模型基础

```bash
python lesson_11_12_tree_models/00_decision_tree_random_forest.py
python lesson_11_12_tree_models/01_feature_importance.py
```

## 课时 13-14：GBDT / XGBoost / LightGBM

```bash
python lesson_13_14_gbdt_xgboost_lightgbm/00_sklearn_gbdt_baseline.py
python lesson_13_14_gbdt_xgboost_lightgbm/01_xgboost_template.py
python lesson_13_14_gbdt_xgboost_lightgbm/02_lightgbm_template.py
```

## 课时 15-16：传统模型工程优化与决策优化

```bash
python lesson_15_16_ml_engineering/00_threshold_profit_optimization.py
python lesson_15_16_ml_engineering/01_probability_calibration.py
```

## 课时 17-18：深度学习基础与全连接网络

```bash
python lesson_17_18_deep_learning_mlp/00_pytorch_mlp_breast_cancer.py
```

## 课时 19-20：CNN 卷积神经网络

```bash
python lesson_19_20_cnn/00_small_cnn_forward.py
python lesson_19_20_cnn/01_cnn_shape_check.py
```

## 课时 21-22：RNN / LSTM / GRU

```bash
python lesson_21_22_rnn_lstm_gru/00_gradient_clipping_demo.py
python lesson_21_22_rnn_lstm_gru/01_gru_sequence_classification.py
```

## 课时 23-24：CNN / RNN 对比复盘

```bash
python lesson_23_24_model_evolution/00_mlp_vs_cnn_input.py
python lesson_23_24_model_evolution/01_rnn_vs_window_features.py
```

## 课时 25-26：Transformer 核心架构与自注意力

```bash
python lesson_25_26_transformer_attention/00_scaled_dot_product_attention.py
python lesson_25_26_transformer_attention/01_sinusoidal_position_encoding.py
python lesson_25_26_transformer_attention/02_pytorch_multihead_attention.py
```

## 课时 27-28：Encoder / Decoder 架构

```bash
python lesson_27_28_encoder_decoder/00_encoder_text_classification.py
python lesson_27_28_encoder_decoder/01_decoder_text_generation.py
python lesson_27_28_encoder_decoder/02_encoder_decoder_translation.py
python lesson_27_28_encoder_decoder/03_model_output_shapes.py
```

## 课时 29-30：大模型预训练逻辑

```bash
python lesson_29_30_pretraining/00_cross_entropy_by_hand.py
python lesson_29_30_pretraining/01_causal_lm_label_shift.py
python lesson_29_30_pretraining/02_context_pooling_embeddings.py
python lesson_29_30_pretraining/03_hallucination_test_cases.py
```

## 课时 31-32：大模型对齐技术

```bash
python lesson_31_32_alignment/00_create_sft_jsonl.py
python lesson_31_32_alignment/01_pairwise_reward_loss.py
python lesson_31_32_alignment/02_answer_loss_mask.py
python lesson_31_32_alignment/03_preference_record.py
```

## 课时 33-34：LoRA 轻量化微调

```bash
python lesson_33_34_lora/00_list_lora_target_modules.py
python lesson_33_34_lora/01_prepare_ticket_dataset.py
python lesson_33_34_lora/02_train_lora_ticket_adapter.py
python lesson_33_34_lora/03_lora_inference.py
python lesson_33_34_lora/04_merge_lora_adapter.py
python lesson_33_34_lora/05_json_evaluator.py
```

## 课时 35-36：大模型工业落地

```bash
python lesson_35_36_llm_production/00_latency_percentiles.py
python lesson_35_36_llm_production/01_simple_router.py
```

## 课时 37-38：完整综合项目

```bash
python lesson_37_38_capstone_project/00_xgboost_pipeline.py
python lesson_37_38_capstone_project/01_choose_threshold.py
python lesson_37_38_capstone_project/02_parse_llm_output.py
```

如果在没有图形界面的环境中运行，部分画图脚本会把图保存到对应课时目录的 `outputs/` 子目录。`outputs/` 已被 `.gitignore` 忽略。
