"""
main.py
Runs:
- Grid input
- DFS Tree
- DFS Graph
- Metric printing
- UI animation
"""

from grid_input import user_input_grid, inflate_obstacles
from dfs_tree import dfs_tree_iterative
from dfs_graph import dfs_graph_iterative, measure
from visualize_ui import GridAnimator
import json
import os


def main():

    grid, start, goal = user_input_grid()
    radius = int(input("Robot radius (cells): "))

    inflated = inflate_obstacles(grid, radius)

    # TREE SEARCH
    tree = dfs_tree_iterative(inflated, start, goal)
    # GRAPH SEARCH
    graph_meas = measure(dfs_graph_iterative, inflated, start, goal)
    graph = graph_meas["result"]

    print("\n=== DFS TREE-SEARCH ===")
    print("Status:", tree["status"])
    print("Found :", tree["found"])
    print("Path length:", len(tree["path"]) if tree["path"] else None)
    print("Explored:", len(tree["explored"]))
    print("Expansions:", tree["expansions"])

    print("\n=== DFS GRAPH-SEARCH ===")
    print("Status:", graph["status"])
    print("Found :", graph["found"])
    print("Visited:", len(graph["visited"]))
    print("Path length:", len(graph["path"]) if graph["path"] else None)
    print("Explored:", len(graph["explored"]))
    print("Expansions:", graph["expansions"])
    print("Time (s):", graph_meas["time_s"])
    print("Memory (KB):", graph_meas["peak_memory_kb"])

    # Launch UI
    path = graph["path"] if graph["found"] else []
    GridAnimator(inflated, start, goal, path, graph["explored"])


if __name__ == "__main__":
    main()
