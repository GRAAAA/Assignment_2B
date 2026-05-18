"""
bfs.py — Breadth-First Search (BFS)

Algorithm type : Uninformed search
Optimal        : Yes — finds the path with the fewest edges (hops) from origin to goal.
                 NOTE: NOT optimal by edge cost — only optimal by hop count.
Complete       : Yes — will always find a solution if one exists on a finite graph.
Time           : O(b^d) where b = branching factor, d = depth of shallowest goal.
Space          : O(b^d) — must keep all frontier nodes in memory (the key BFS drawback).

Tiebreaking (required by spec NOTE 2):
    When multiple neighbours exist, enqueue in ASCENDING NODE ID order.
    Nodes enqueued earlier are dequeued first (FIFO), satisfying the chronological
    ordering rule automatically: N1 added before N2 → N1 expanded before N2.
"""

from collections import deque


def bfs(graph, coordinates, origin, destinations):
    """
    Breadth-First Search.

    Explores all nodes at depth d before exploring any node at depth d+1.
    Guaranteed to find the path with the fewest edges to a goal.

    Args:
        graph        (dict): Adjacency list.
                             { node_id (int): [(neighbour_id (int), cost (int/float)), ...] }
                             Edges are DIRECTED.
        coordinates  (dict): Node 2D positions — not used by BFS (uninformed), kept for
                             interface compatibility.
        origin        (int): Starting node ID.
        destinations  (list): List of goal node IDs. Reaching ANY ONE is success.

    Returns:
        tuple: (goal_node, nodes_created, path)
            goal_node     (int)  : The destination node reached.
            nodes_created  (int)  : Total nodes generated (every enqueue = +1),
                                    including the origin.
            path          (list) : Node IDs from origin to goal, inclusive.

        If no path exists:
            (None, nodes_created, [])
    """

    dest_set = set(destinations)

    # ── Early exit: origin is already a destination ───────────────────────────
    if origin in dest_set:
        return origin, 1, [origin]

    # ── Queue (FIFO frontier) ─────────────────────────────────────────────────
    # Each entry: (node_id, path_so_far)
    queue = deque([(origin, [origin])])

    # ── Explored set ──────────────────────────────────────────────────────────
    # For BFS, we mark nodes as explored at ENQUEUE time (not dequeue time).
    # This is the correct BFS approach: it prevents the same node from being
    # added to the queue multiple times, which would waste work and miscount nodes.
    # Because BFS always finds the shallowest path first, the first time we reach
    # a node is guaranteed to be via the shortest-hop path.
    explored = {origin}

    # ── Node creation counter ─────────────────────────────────────────────────
    # Count every node enqueued. Origin counts as 1 (enqueued at start).
    nodes_created = 1

    # ── Main search loop ──────────────────────────────────────────────────────
    while queue:

        current, path = queue.popleft()

        # ── Goal test on expansion ────────────────────────────────────────────
        if current in dest_set:
            return current, nodes_created, path

        # ── Expand: generate successors ───────────────────────────────────────
        # Sort neighbours by node ID ascending (spec NOTE 2: ascending tiebreak).
        # FIFO queue ensures nodes enqueued earlier are expanded first, satisfying
        # the chronological ordering rule automatically.
        neighbours = sorted(graph.get(current, []), key=lambda edge: edge[0])

        for neighbour_id, _edge_cost in neighbours:
            if neighbour_id not in explored:
                explored.add(neighbour_id)  # mark at enqueue time to prevent duplicates
                nodes_created += 1
                queue.append((neighbour_id, path + [neighbour_id]))

    # ── No path found ─────────────────────────────────────────────────────────
    return None, nodes_created, []
