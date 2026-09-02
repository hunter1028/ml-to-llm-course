"""课时 31-32：手工偏好标注记录。"""

import json


preference = {
    "prompt": "根据给定材料回答，不要补充材料外事实。",
    "chosen": "材料中没有说明具体日期，因此无法确定。",
    "rejected": "该事件发生在2023年5月。",
    "reasons": ["忠实于材料", "没有虚构日期"],
    "annotator_confidence": 0.95,
    "policy_version": "v1",
}


def main() -> None:
    print(json.dumps(preference, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
