# main.py
"""
Main runner: uses grid_input, dfs_tree, dfs_graph, robot_sim, visualize_ui.
Saves metrics (CSV and JSON) and visualizations.
"""

import os
import json
from grid_input import user_input_grid, inflate_obstacles
from dfs_tree import dfs_tree_iterative
from dfs_graph import dfs_graph_iterative, measure as measure_graph
from visualize_dual_ui import DualGridAnimator
from robot_sim import animate_and_save
import pandas as pd
from robot_sim import animate_path_live


OUT_DIR = "dfs_outputs"


def save_metrics(metrics_list, out_dir=OUT_DIR):
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


def save_static_image(grid, start, goal, explored, path, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6))
    img = np.ones((grid.shape[0], grid.shape[1], 3), dtype=float)
    img[grid == 1] = (0.7, 0.7, 0.7)
    ax.imshow(img, origin="upper")
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    if explored:
        ex_r = [p[0] for p in explored]
        ex_c = [p[1] for p in explored]
        ax.scatter(ex_c, ex_r, marker="s", s=80, facecolors="none", edgecolors="magenta")
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


def main():
    grid, start, goal = user_input_grid()
    # robot radius in cells
    while True:
        try:
            r = int(input("Robot radius (cells, 0 for point): ").strip())
            if r < 0:
                print("Radius must be >= 0")
                continue
            break
        except:
            print("Invalid integer. Example: 0 or 1")

    # inflate obstacles
    inflated = inflate_obstacles(grid, r, metric="euclidean")
    os.makedirs(OUT_DIR, exist_ok=True)
    # save inflated grid for inspection
    import numpy as np
    np.savetxt(os.path.join(OUT_DIR, "inflated_grid.txt"), inflated, fmt="%d")

    # run tree DFS (measured using time/tracemalloc inside function if you want; here we do simple run)
    import time, tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    tree_res = dfs_tree_iterative(inflated, start, goal, order="URDL", max_depth=None, max_expansions=None)
    t1 = time.perf_counter()
    c, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tree_meas = {"result": tree_res, "time_s": t1 - t0, "peak_memory_kb": peak / 1024.0}

    # run graph DFS (use provided measure wrapper)
    graph_meas = measure_graph(dfs_graph_iterative, inflated, start, goal, "URDL")
    graph_res = graph_meas["result"]

    # Prepare metrics list for saving
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
            "time_s": tree_meas["time_s"],
            "peak_memory_kb": tree_meas["peak_memory_kb"],
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
            "time_s": graph_meas["time_s"],
            "peak_memory_kb": graph_meas["peak_memory_kb"],
        },
    ]

    # print metrics in terminal
    print("\n=== DFS TREE-SEARCH METRICS ===")
    print(f"Status        : {tree_res.get('status')}")
    print(f"Found         : {tree_res.get('found')}")
    print(f"Path length   : {len(tree_res['path']) if tree_res.get('path') else 'N/A'}")
    print(f"Explored nodes: {len(tree_res.get('explored', []))}")
    print(f"Expansions    : {tree_res.get('expansions')}")
    print(f"Time (s)      : {tree_meas['time_s']:.6f}")
    print(f"Peak memory KB: {tree_meas['peak_memory_kb']:.2f}")

    print("\n=== DFS GRAPH-SEARCH METRICS ===")
    print(f"Status        : {graph_res.get('status')}")
    print(f"Found         : {graph_res.get('found')}")
    print(f"Path length   : {len(graph_res['path']) if graph_res.get('path') else 'N/A'}")
    print(f"Explored nodes: {len(graph_res.get('explored', []))}")
    print(f"Visited nodes : {len(graph_res.get('visited', []))}")
    print(f"Expansions    : {graph_res.get('expansions')}")
    print(f"Time (s)      : {graph_meas['time_s']:.6f}")
    print(f"Peak memory KB: {graph_meas['peak_memory_kb']:.2f}")

    # Save metrics (CSV + JSON)
    save_metrics(metrics, OUT_DIR)

    # Save static figures for both algorithms
    save_static_image(inflated, start, goal, tree_res.get("explored"), tree_res.get("path"), os.path.join(OUT_DIR, "dfs_tree.png"))
    save_static_image(inflated, start, goal, graph_res.get("explored"), graph_res.get("path"), os.path.join(OUT_DIR, "dfs_graph.png"))
    print("Saved static images to", OUT_DIR)

    # Animate graph path and save mp4/frames
    sim_path = graph_res.get("path") if graph_res.get("found") else (tree_res.get("path") if tree_res.get("found") else None)
    if sim_path:
        animate_path_live(inflated, sim_path, start, goal)
    else:
        print("No path found by either DFS variant; skipping animation.")

    # launch UI (animates graph path)
    tree_explored = tree_res.get("explored") or []
    tree_path = tree_res.get("path") or []
    graph_explored = graph_res.get("explored") or []
    graph_path = graph_res.get("path") or []

    DualGridAnimator(inflated, start, goal, tree_explored, tree_path, graph_explored, graph_path)


if __name__ == "__main__":
    main()
