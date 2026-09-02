"""课时 29-30：设计幻觉测试用例表。"""

import json


cases = [
    {
        "question": "询问可从权威资料核验的事实",
        "expected_source": "指定数据库或官方网站",
        "risk": "虚构日期或来源",
    },
    {
        "question": "包含错误前提的问题",
        "expected_source": "应先纠正前提",
        "risk": "顺着错误前提继续编造",
    },
]


def main() -> None:
    print(json.dumps(cases, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
