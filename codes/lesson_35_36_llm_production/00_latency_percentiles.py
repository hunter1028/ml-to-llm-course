"""课时 35-36：统计模型调用延迟分位数。"""

import time

import numpy as np


def fake_call_model(sample: dict) -> dict:
    """本地模拟一次模型调用，真实项目中替换为 API 或服务调用。"""
    time.sleep(sample["sleep_ms"] / 1000)
    return {
        "ok": True,
        "input_tokens": sample["input_tokens"],
        "output_tokens": sample["output_tokens"],
    }


def benchmark(call_model, samples, concurrency_label="single"):
    records = []
    for sample in samples:
        start = time.perf_counter()
        result = call_model(sample)
        elapsed_ms = (time.perf_counter() - start) * 1000
        records.append({
            "elapsed_ms": elapsed_ms,
            "ok": result.get("ok", False),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        })

    latencies = np.array([r["elapsed_ms"] for r in records])
    return {
        "mode": concurrency_label,
        "count": len(records),
        "success_rate": np.mean([r["ok"] for r in records]),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in records),
    }


def main() -> None:
    rng = np.random.default_rng(42)
    samples = [
        {
            "sleep_ms": float(rng.uniform(5, 30)),
            "input_tokens": int(rng.integers(80, 300)),
            "output_tokens": int(rng.integers(20, 120)),
        }
        for _ in range(30)
    ]
    result = benchmark(fake_call_model, samples)
    for k, v in result.items():
        print(k, round(v, 2) if isinstance(v, float) else v)


if __name__ == "__main__":
    main()
