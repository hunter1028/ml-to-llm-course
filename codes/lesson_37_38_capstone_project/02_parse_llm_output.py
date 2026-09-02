"""课时 37-38：解析并校验大模型 JSON 输出。"""

import json


ALLOWED_RISKS = {"safety", "reputation", "repeat", "amount", "unknown"}


def parse_llm_output(text):
    obj = json.loads(text)
    required = {"need_escalation", "risk_type", "reason", "suggested_action"}
    if not required.issubset(obj):
        raise ValueError("missing fields")
    if obj["risk_type"] not in ALLOWED_RISKS:
        raise ValueError("invalid risk_type")
    if not isinstance(obj["need_escalation"], bool):
        raise TypeError("need_escalation must be boolean")
    return obj


def main() -> None:
    text = json.dumps({
        "need_escalation": True,
        "risk_type": "safety",
        "reason": "用户描述充电器发热并有焦味，存在安全风险。",
        "suggested_action": "转人工并建立安全工单。",
    }, ensure_ascii=False)
    print(parse_llm_output(text))


if __name__ == "__main__":
    main()
