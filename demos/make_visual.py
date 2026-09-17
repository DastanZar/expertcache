"""Animate real online cache transitions; rectangle widths encode byte sizes."""

import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expertcache.core import Residency  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sizes = {0: 2, 1: 3, 2: 4, 3: 2, 4: 4, 5: 3}
trace = [0, 1, 0, 1, 2, 3, 4, 5, 0, 1]
states = []
caches = [Residency(sizes, 10, p) for p in ["lru", "cost2q"]]
for key in trace:
    frame = []
    for cache in caches:
        before = cache.misses
        with cache.lease([key]):
            frame.append((list(cache.resident), cache.misses, cache.misses > before))
    states.append(frame)
colors = ["#c2410c", "#0f766e", "#a16207", "#7c3aed", "#2563eb", "#be123c"]
fig, ax = plt.subplots(figsize=(11, 5), facecolor="#fff7ed")


def draw(index: int) -> None:
    ax.clear()
    ax.set_facecolor("#fff7ed")
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.text(
        0,
        4,
        "expertcache / residency is a byte budget, not a slot count",
        fontsize=17,
        weight="bold",
        color="#431407",
    )
    key = trace[index]
    ax.text(
        0,
        3.3,
        f"Request {index + 1:02d}: expert {key} • payload {sizes[key]} units",
        fontsize=14,
        color="#9a3412",
    )
    for row, (policy, state) in enumerate(zip(["LRU", "COST2Q"], states[index], strict=True)):
        residents, misses, miss = state
        y = 1.6 - row * 1.7
        ax.text(
            0,
            y + 0.85,
            f"{policy}   |   {'LOAD' if miss else 'HIT'}   |   misses {misses}",
            fontsize=12,
            weight="bold",
            color="#334155",
        )
        ax.add_patch(Rectangle((0, y), 10, 0.65, fill=False, ec="#9a3412", lw=1.5))
        x = 0
        for expert in residents:
            width = sizes[expert]
            ax.add_patch(Rectangle((x, y), width, 0.65, fc=colors[expert], ec="#fff7ed", lw=2))
            ax.text(
                x + width / 2,
                y + 0.325,
                f"E{expert} · {width}",
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=12,
            )
            x += width
        ax.text(10.2, y + 0.325, f"{x}/10", va="center", fontsize=10)
    fig.texts.clear()
    fig.text(
        0.08,
        0.025,
        "Two-hit promotion protects reuse from scans. All choices use past accesses only.",
        color="#7c2d12",
        fontsize=11,
    )


out = ROOT / "assets"
out.mkdir(exist_ok=True)
animation = FuncAnimation(fig, draw, frames=len(states), interval=950, repeat=True)
animation.save(out / "residency.gif", writer=PillowWriter(fps=1))
draw(9)
fig.savefig(out / "residency.png", dpi=160)
print("ANIMATION_OK: 10 frames from Residency.lease")
