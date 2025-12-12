# main.py
"""
Main runner: uses grid_input, dfs_tree, dfs_graph, visualize_dual_ui.
Saves metrics (CSV + JSON) and static images.
Launches ONLY the DualGridAnimator (dual UI).
"""

import os
import json
import time
import tracemalloc
from typing import List, Dict

import numpy as np
import pandas as pd

from grid_input import user_input_grid, inflate_obstacles
from dfs_tree import dfs_tree_iterative
from dfs_graph import dfs_graph_iterative, measure as measure_graph
from visualize_dual_ui import DualGridAnimator

OUT_DIR = "dfs_outputs"


def save_metrics(metrics_list: List[Dict], out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df_rows = []
    for m in metrics_list:
        df_rows.append(
            {
                "algorithm": m.get("algorithm"),
                "variant": m.get("variant"),
                "found": m.get("found"),
                "time_s": m.get("time_s"),
                "peak_memory_kb": m.get("peak_memory_kb"),
                "explored_count": len(m.get("explored")) if m.get("explored") else None,
                "visited_count": len(m.get("visited")) if m.get("visited") else None,
                "path_length": len(m.get("path")) if m.get("path") else None,
                "expansions": m.get("expansions"),
                "status": m.get("status"),
            }
        )
    df = pd.DataFrame(df_rows)
    csv_path = os.path.join(out_dir, "dfs_metrics.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "dfs_verbose.json"), "w") as f:
        json.dump(metrics_list, f, indent=2)
    print("Saved metrics to", csv_path)


def save_static_image(grid: np.ndarray, start: tuple, goal: tuple, explored: List[tuple], path: List[tuple], filename: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as _np

    fig, ax = plt.subplots(figsize=(6, 6))
    img = _np.ones((grid.shape[0], grid.shape[1], 3), dtype=float)
    img[grid == 1] = (0.7, 0.7, 0.7)
    ax.imshow(img, origin="upper")
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)

    # Show exploration (magenta)
    if explored:
        ex_r = [p[0] for p in explored]
        ex_c = [p[1] for p in explored]
        ax.scatter(ex_c, ex_r, marker="s", s=80, facecolors="none", edgecolors="magenta")

    # Optionally show final path in static image (this does not open UI)
    if path:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, "-k", linewidth=3)
        ax.scatter(pc, pr, s=60, facecolors="yellow", edgecolors="k")

    ax.scatter([start[1]], [start[0]], c="red", s=140, marker="s", edgecolors="k")
    ax.scatter([goal[1]], [goal[0]], c="green", s=140, marker="s", edgecolors="k")

    plt.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def print_metrics_block(title: str, res: Dict, time_s: float, peak_kb: float) -> None:
    print(f"\n=== {title} ===")
    print(f"Status        : {res.get('status')}")
    print(f"Found         : {res.get('found')}")
    print(f"Path length   : {len(res['path']) if res.get('path') else 'N/A'}")
    print(f"Explored nodes: {len(res.get('explored', []))}")
    if res.get("visited") is not None:
        try:
            visited_len = len(res.get("visited"))
        except Exception:
            visited_len = "N/A"
        print(f"Visited nodes : {visited_len}")
    print(f"Expansions    : {res.get('expansions')}")
    print(f"Time (s)      : {time_s:.6f}")
    print(f"Peak memory KB: {peak_kb:.2f}")


def main():
    # 1) get grid, start, goal
    grid, start, goal = user_input_grid()

    # 2) robot radius
    while True:
        try:
            r = int(input("Robot radius (cells, 0 for point): ").strip())
            if r < 0:
                print("Radius must be >= 0")
                continue
            break
        except Exception:
            print("Invalid integer. Example: 0 or 1")

    # 3) inflate obstacles and save grid
    inflated = inflate_obstacles(grid, r, metric="euclidean")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savetxt(os.path.join(OUT_DIR, "inflated_grid.txt"), inflated, fmt="%d")
    print("Saved inflated grid to", os.path.join(OUT_DIR, "inflated_grid.txt"))

    # 4) run TREE DFS (timing + tracemalloc)
    tracemalloc.start()
    t0 = time.perf_counter()
    tree_res = dfs_tree_iterative(inflated, start, goal, order="URDL", max_depth=None, max_expansions=None)
    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tree_time = t1 - t0
    tree_peak_kb = peak / 1024.0

    # 5) run GRAPH DFS (use measure wrapper)
    graph_meas = measure_graph(dfs_graph_iterative, inflated, start, goal, "URDL")
    graph_res = graph_meas["result"]
    graph_time = graph_meas["time_s"]
    graph_peak_kb = graph_meas["peak_memory_kb"]

    # 6) prepare metrics
    metrics = [
        {
            "algorithm": "DFS",
            "variant": "tree-search",
            "found": tree_res.get("found"),
            "path": tree_res.get("path"),
            "explored": tree_res.get("explored"),
            "visited": None,
            "expansions": tree_res.get("expansions"),
            "status": tree_res.get("status"),
            "time_s": tree_time,
            "peak_memory_kb": tree_peak_kb,
        },
        {
            "algorithm": "DFS",
            "variant": "graph-search",
            "found": graph_res.get("found"),
            "path": graph_res.get("path"),
            "explored": graph_res.get("explored"),
            "visited": list(graph_res.get("visited")) if graph_res.get("visited") is not None else None,
            "expansions": graph_res.get("expansions"),
            "status": graph_res.get("status"),
            "time_s": graph_time,
            "peak_memory_kb": graph_peak_kb,
        },
    ]

    # 7) print metrics
    print_metrics_block("DFS TREE-SEARCH METRICS", tree_res, tree_time, tree_peak_kb)
    print_metrics_block("DFS GRAPH-SEARCH METRICS", graph_res, graph_time, graph_peak_kb)

    # 8) save metrics and static images


    # 9) launch only the DualGridAnimator (dual UI)
    tree_explored = tree_res.get("explored") or []
    tree_path = tree_res.get("path") or []
    graph_explored = graph_res.get("explored") or []
    graph_path = graph_res.get("path") or []

    DualGridAnimator(inflated, start, goal, tree_explored, tree_path, graph_explored, graph_path, delay=0.18)


if __name__ == "__main__":
    main()
