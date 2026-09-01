"""课程示例代码入口。

完整示例按课时放在子目录中：
- lesson_00_intro
- lesson_07_08_linear_models

运行具体脚本请看 README.md。
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("课程示例代码已按课时分目录存放。")
    print(f"目录：{ROOT}")
    print("运行方式示例：")
    print("  python lesson_07_08_linear_models/00_normal_equation_demo.py")
    print("  python lesson_07_08_linear_models/01_linear_regression_diabetes.py")
    print("  python lesson_07_08_linear_models/02_logistic_regression_breast_cancer.py")
    print("  python lesson_07_08_linear_models/03_l1_l2_huber_regression.py")


if __name__ == "__main__":
    main()
