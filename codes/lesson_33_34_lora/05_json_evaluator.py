"""课时 33-34：最小 JSON 评测器。"""

import json


REQUIRED = {"intent", "urgency", "order_id", "action"}


def evaluate_json(text: str, expected: dict) -> dict:
    result = {
        "json_valid": False,
        "schema_valid": False,
        "field_correct": 0,
    }
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return result

    result["json_valid"] = True
    result["schema_valid"] = REQUIRED.issubset(obj)
    result["field_correct"] = sum(
        obj.get(key) == expected.get(key) for key in REQUIRED
    )
    return result


def main() -> None:
    pred = '{"intent":"质量问题","urgency":"中","order_id":"A1024","action":"创建换货工单"}'
    expected = {
        "intent": "质量问题",
        "urgency": "中",
        "order_id": "A1024",
        "action": "创建换货工单",
    }
    print(evaluate_json(pred, expected))


if __name__ == "__main__":
    main()
