"""Synthetic online routing traces; actual CPU expert copy and execution."""

import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expertcache.execute import Engine, make_bank, resident_forward, sizes_of  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def traces() -> dict[str, list[list[int]]]:
    rng = random.Random(41)
    hot_scan = []
    for _ in range(16):
        hot_scan.extend([[0, 1]] * 3 + [[i, i + 1] for i in range(2, 14, 2)] + [[0, 1]] * 2)
    uniform = [rng.sample(range(14), 2) for _ in range(len(hot_scan))]
    drift = [rng.sample(range(base, base + 5), 2) for base in [0, 7, 0, 7] for _ in range(44)]
    return {"hot_scan": hot_scan, "uniform": uniform, "phase_shift": drift}


def main() -> None:
    torch.set_num_threads(1)
    bank = make_bank([16, 20, 24, 32, 40, 48, 16, 24, 32, 40, 48, 20, 32, 40])
    sizes = sizes_of(bank)
    x = torch.linspace(-1, 1, 16).reshape(1, 16)
    rows = []
    for name, trace in traces().items():
        reference = [resident_forward(bank, x, route, [0.4, 0.6]) for route in trace]
        for budget in [sum(sorted(sizes.values())[-3:]), sum(sorted(sizes.values())[-5:])]:
            for policy in ["lru", "cost2q"]:
                samples = []
                resident_samples = []
                for rep in range(5):
                    engine = Engine(bank, budget, policy)
                    outputs = []
                    start = time.perf_counter_ns()
                    for route in trace:
                        outputs.append(engine.forward(x, route, [0.4, 0.6]))
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    start = time.perf_counter_ns()
                    for route in trace:
                        resident_forward(bank, x, route, [0.4, 0.6])
                    resident_ms = (time.perf_counter_ns() - start) / 1e6
                    for output, expected in zip(outputs, reference, strict=True):
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    assert engine.cache.bytes_loaded == engine.actual_bytes_copied
                    if rep:
                        samples.append(elapsed)
                        resident_samples.append(resident_ms)
                cache = engine.cache
                # A serial link estimate only: never mislabeled as measured PCIe.
                modeled_ms = (cache.bytes_loaded / 12e9 + cache.misses * 5e-6) * 1e3
                rows.append(
                    dict(
                        trace=name,
                        policy=policy,
                        capacity_bytes=budget,
                        requests=len(trace),
                        expert_uses=2 * len(trace),
                        misses=cache.misses,
                        hits=cache.hits,
                        bytes_loaded=cache.bytes_loaded,
                        evictions=cache.evictions,
                        final_resident_bytes=cache.used,
                        cpu_ms_samples=samples,
                        median_cpu_ms=statistics.median(samples),
                        resident_ms_samples=resident_samples,
                        median_resident_ms=statistics.median(resident_samples),
                        modeled_serial_link_ms=modeled_ms,
                        exact_output_parity=True,
                    )
                )
    receipt = dict(
        schema=1,
        device="CPU",
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        threads=1,
        seed=41,
        sizes_bytes=sizes,
        backing_bank_bytes=sum(sizes.values()),
        warmup=1,
        repetitions=4,
        weights=[0.4, 0.6],
        traces=traces(),
        transfer_model={"bandwidth_bytes_s": 12e9, "setup_seconds": 5e-6, "measured": False},
        timing="CPU wall clock includes policy, deep copy, expert execution; excludes parity",
        cases=rows,
    )
    out = ROOT / "receipts"
    out.mkdir(exist_ok=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#fff7ed")
    for ax, name in zip(axes, traces(), strict=True):
        subset = [r for r in rows if r["trace"] == name]
        for policy, color in [("lru", "#64748b"), ("cost2q", "#c2410c")]:
            arm = [r for r in subset if r["policy"] == policy]
            ax.plot(
                [r["capacity_bytes"] / 1024 for r in arm],
                [r["bytes_loaded"] / 1024 for r in arm],
                "o-",
                label=policy,
                color=color,
            )
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Residency budget (KiB)")
        ax.set_ylabel("Actual copied parameter payload (KiB)")
        ax.legend()
    fig.suptitle("No future oracle: reuse protection trades off with phase changes")
    fig.tight_layout()
    fig.savefig(out / "traffic.png", dpi=160)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "trace",
                        "policy",
                        "capacity_bytes",
                        "misses",
                        "bytes_loaded",
                        "median_cpu_ms",
                    ]
                }
                for r in rows
            ],
            indent=2,
        )
    )
    assert all(r["exact_output_parity"] for r in rows)
    print("PARITY_OK")


if __name__ == "__main__":
    main()
