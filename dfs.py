"""
dfs.py — Depth-First Search (DFS)

Algorithm type : Uninformed search
Optimal        : No  — follows the deepest branch first, may return a longer path
Complete       : Yes — will find a solution if one exists (on finite graphs with explored tracking)
Time           : O(b^m) where b = branching factor, m = maximum depth
Space          : O(b*m) — only the current path and its siblings need to be remembered

Tiebreaking (required by spec NOTE 2):
    When multiple neighbours exist, expand in ASCENDING NODE ID order.
    Enforced by sorting neighbours descending before pushing onto the stack —
    so the smallest-ID neighbour ends up on top and is popped (expanded) first.
    Chronological order among nodes on different branches is satisfied naturally
    by the LIFO stack: the most recently pushed branch is explored first, which
    corresponds to the last parent's children being explored before backtracking.
"""

def dfs(graph, coordinates, origin, destinations):
    """
    Depth-First Search.

    Explores as deep as possible along each branch before backtracking.
    Always expands the most recently discovered unexplored node.

    Args:
        graph        (dict): Adjacency list.
                             { node_id (int): [(neighbour_id (int), cost (int/float)), ...] }
                             Edges are DIRECTED.
        coordinates  (dict): Node 2D positions — not used by DFS (uninformed), kept for
                             interface compatibility.
        origin        (int): Starting node ID.
        destinations  (list): List of goal node IDs. Reaching ANY ONE is success.

    Returns:
        tuple: (goal_node, nodes_created, path)
            goal_node     (int)  : The destination node reached.
            nodes_created  (int)  : Total nodes generated (every stack push = +1),
                                    including the origin.
            path          (list) : Node IDs from origin to goal, inclusive.

        If no path exists:
            (None, nodes_created, [])
    """

    dest_set = set(destinations)

    # ── Early exit: origin is already a destination ───────────────────────────
    if origin in dest_set:
        return origin, 1, [origin]

    # ── Stack (LIFO frontier) ─────────────────────────────────────────────────
    # Each entry: (node_id, path_so_far)
    # We use lazy deletion: the same node can appear multiple times on the stack.
    # When popped, we check the explored set and skip if already expanded.
    stack = [(origin, [origin])]

    # ── Explored set (closed list) ────────────────────────────────────────────
    explored = set()

    # ── Node creation counter ─────────────────────────────────────────────────
    # Count every node pushed onto the stack. Origin is the first node created.
    nodes_created = 1

    # ── Main search loop ──────────────────────────────────────────────────────
    while stack:

        current, path = stack.pop()

        # Lazy deletion: skip nodes that were already expanded via a better path
        if current in explored:
            continue

        explored.add(current)

        # ── Goal test on expansion ────────────────────────────────────────────
        if current in dest_set:
            return current, nodes_created, path

        # ── Expand: generate successors ───────────────────────────────────────
        # Sort neighbours by node ID DESCENDING before pushing, so the
        # smallest-ID neighbour sits on top of the stack and is explored first.
        # This satisfies spec NOTE 2: ascending expansion order.
        neighbours = sorted(graph.get(current, []), key=lambda edge: edge[0], reverse=True)

        for neighbour_id, _edge_cost in neighbours:
            if neighbour_id not in explored:
                nodes_created += 1
                stack.append((neighbour_id, path + [neighbour_id]))

    # ── No path found ─────────────────────────────────────────────────────────
    return None, nodes_created, []
