"""课时 35-36：大模型生产系统中的简单路由器。"""


class SmallModel:
    def predict(self, ticket: dict) -> dict:
        if "退款" in ticket["text"] and not ticket["is_long_text"]:
            return {"label": "refund", "confidence": 0.93}
        return {"label": "unknown", "confidence": 0.62}


def route(ticket, small_model):
    pred = small_model.predict(ticket)

    if ticket["has_safety_risk"]:
        return "human_review"

    if pred["confidence"] >= 0.90 and not ticket["is_long_text"]:
        return "traditional_model"

    if ticket["needs_policy_lookup"]:
        return "rag_llm"

    return "llm"


def main() -> None:
    small_model = SmallModel()
    tickets = [
        {
            "text": "申请退款，订单 A1001。",
            "has_safety_risk": False,
            "is_long_text": False,
            "needs_policy_lookup": False,
        },
        {
            "text": "产品疑似安全风险，需要处理。",
            "has_safety_risk": True,
            "is_long_text": False,
            "needs_policy_lookup": False,
        },
        {
            "text": "这类保修规则是什么？",
            "has_safety_risk": False,
            "is_long_text": False,
            "needs_policy_lookup": True,
        },
    ]

    for ticket in tickets:
        print(ticket["text"], "=>", route(ticket, small_model))


if __name__ == "__main__":
    main()
