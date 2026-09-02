"""课程示例代码入口。

完整示例按课时放在子目录中。运行具体脚本请看 README.md。
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

LESSON_DIRS = [
    ("lesson_00_intro", "两个小 AI 例子"),
    ("lesson_01_02_ai_intro", "课时 01-02：行业认知与学习方法论"),
    ("lesson_03_04_math_foundations", "课时 03-04：AI 核心数学基础"),
    ("lesson_05_06_ml_basics", "课时 05-06：机器学习基础范式与通用原理"),
    ("lesson_07_08_linear_models", "课时 07-08：线性模型理论与实战"),
    ("lesson_09_10_clustering_pca", "课时 09-10：聚类与降维"),
    ("lesson_11_12_tree_models", "课时 11-12：树模型"),
    ("lesson_13_14_gbdt_xgboost_lightgbm", "课时 13-14：GBDT / XGBoost / LightGBM"),
    ("lesson_15_16_ml_engineering", "课时 15-16：模型工程与决策优化"),
    ("lesson_17_18_deep_learning_mlp", "课时 17-18：深度学习与全连接网络"),
    ("lesson_19_20_cnn", "课时 19-20：CNN"),
    ("lesson_21_22_rnn_lstm_gru", "课时 21-22：RNN / LSTM / GRU"),
    ("lesson_23_24_model_evolution", "课时 23-24：CNN / RNN 对比复盘"),
    ("lesson_25_26_transformer_attention", "课时 25-26：Transformer 与自注意力"),
    ("lesson_27_28_encoder_decoder", "课时 27-28：Encoder / Decoder"),
    ("lesson_29_30_pretraining", "课时 29-30：大模型预训练"),
    ("lesson_31_32_alignment", "课时 31-32：SFT / RM / RLHF"),
    ("lesson_33_34_lora", "课时 33-34：LoRA 微调"),
    ("lesson_35_36_llm_production", "课时 35-36：大模型工业落地"),
    ("lesson_37_38_capstone_project", "课时 37-38：综合项目"),
]


def main() -> None:
    print("课程示例代码已按课时分目录存放。")
    print(f"目录：{ROOT}")
    print("\n目录清单：")
    for folder, title in LESSON_DIRS:
        print(f"  {folder:<42s} {title}")
    print("\n运行方式示例：")
    print("  cd /Users/liuzhicheng/Documents/01pgm/ml-course/codes")
    print("  python lesson_09_10_clustering_pca/00_kmeans_pca_user_segments.py")
    print("  python lesson_15_16_ml_engineering/00_threshold_profit_optimization.py")
    print("  python lesson_25_26_transformer_attention/00_scaled_dot_product_attention.py")


if __name__ == "__main__":
    main()
