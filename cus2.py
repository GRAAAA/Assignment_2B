"""
cus2.py — CUS2: Iterative Deepening A* (IDA*)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE CHOSE IDA* FOR CUS2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The spec requires CUS2 to be:
  (1) An INFORMED method — uses heuristic/cost information.
  (2) Finds the SHORTEST PATH (with least moves / lowest cost) to the goal.

IDA* (Korf, 1985) is the ideal choice because:

  vs A*:
    - A* stores the entire frontier in memory: O(b^d) space.
    - IDA* achieves the same optimal result using only O(b*d) space —
      a single DFS path — by trading repeated work for memory efficiency.
    - For large search spaces, IDA* is far more practical.

  vs GBFS:
    - GBFS is fast but NOT optimal — it ignores accumulated path cost.
    - IDA* is optimal because it prunes on f(n) = g(n) + h(n), ensuring
      the first solution found is always the lowest-cost one.

  The cost of IDA* is re-expanding nodes at each iteration.
  In practice the overhead is small: because node counts grow exponentially
  with depth, the deepest level dominates and re-expansion adds only
  a constant factor (≈ b/(b-1)) over the total work of A*.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM PROPERTIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Complete : Yes — will find a solution if one exists on a finite graph.
  Optimal  : Yes — guaranteed to find the minimum-cost path, provided the
             heuristic is ADMISSIBLE (Euclidean distance never overestimates).
  Time     : O(b^d) — same asymptotic complexity as A*.
  Space    : O(b * d) — same as DFS, far better than A*'s O(b^d).
  Informed : Yes — uses f(n) = g(n) + h(n), Euclidean distance heuristic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Outer loop: start with bound = h(origin) — the tightest admissible bound.
  Inner call: depth-first search, but PRUNE any node where f(n) > bound.

  If goal found within bound → return it (optimal because h is admissible).
  If not → set bound = minimum f(n) that was pruned, then retry.

  Each iteration uses a tighter bound, converging on the optimal solution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIEBREAKING (spec NOTE 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rule 1 — Ascending node ID: neighbours are sorted by node ID before
            each expansion. Smaller IDs are explored first.
  Rule 2 — Chronological order: the DFS explores neighbours left-to-right
            after sorting, which is naturally chronological.
"""

import math


# ── Helpers ───────────────────────────────────────────────────────────────────

def _euclidean(coord1, coord2):
    """Straight-line distance between two (x, y) coordinate tuples."""
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def _heuristic(node_id, destinations, coordinates):
    """
    h(n) = minimum Euclidean distance from node_id to any destination.

    Admissibility: Euclidean distance is always <= true travel cost along edges
    on a 2D coordinate graph, so this heuristic never overestimates.
    """
    node_coord = coordinates[node_id]
    return min(_euclidean(node_coord, coordinates[d]) for d in destinations)


# ── Recursive DFS with f-bound pruning ───────────────────────────────────────

def _search(graph, coordinates, current, dest_set, destinations,
            g, bound, path, on_path, nodes_created_ref):
    """
    Depth-first search pruned by f(n) = g(n) + h(n) <= bound.

    Args:
        graph             (dict) : Adjacency list.
        coordinates       (dict) : Node 2D positions.
        current            (int) : Node being explored now.
        dest_set           (set) : Set of goal node IDs.
        destinations      (list) : List of goal node IDs (for heuristic).
        g                (float) : Cumulative cost to reach current node.
        bound            (float) : Current f-value pruning threshold.
        path              (list) : Node IDs on the current DFS path.
        on_path            (set) : Same nodes as path, as a set for O(1) lookup.
                                   Used for cycle detection (avoids revisiting
                                   ancestors on the current branch).
        nodes_created_ref (list) : [int] — mutable counter for nodes generated.

    Returns:
        (threshold, goal_node, found_path)
        threshold  — minimum f value pruned; -inf if goal was found.
        goal_node  — int if goal found, else None.
        found_path — list if goal found, else [].
    """

    f = g + _heuristic(current, destinations, coordinates)

    # Prune: f exceeds current bound → return f as candidate for next bound
    if f > bound:
        return f, None, []

    # ── Goal test ─────────────────────────────────────────────────────────────
    if current in dest_set:
        return -math.inf, current, list(path)

    minimum = math.inf

    # ── Expand: generate successors ───────────────────────────────────────────
    # Sort by node ID ascending to satisfy spec tiebreak rule 1.
    neighbours = sorted(graph.get(current, []), key=lambda edge: edge[0])

    for neighbour_id, edge_cost in neighbours:

        # Cycle guard: skip nodes already on the current path
        if neighbour_id in on_path:
            continue

        nodes_created_ref[0] += 1

        on_path.add(neighbour_id)
        path.append(neighbour_id)

        t, found_goal, found_path = _search(
            graph, coordinates, neighbour_id, dest_set, destinations,
            g + edge_cost, bound, path, on_path, nodes_created_ref
        )

        if found_goal is not None:
            return -math.inf, found_goal, found_path

        if t < minimum:
            minimum = t

        path.pop()
        on_path.remove(neighbour_id)

    return minimum, None, []


# ── Main public function ──────────────────────────────────────────────────────

def cus2(graph, coordinates, origin, destinations):
    """
    CUS2: Iterative Deepening A* (IDA*).

    Informed search — uses f(n) = g(n) + h(n) with Euclidean distance heuristic.
    Finds the optimal (lowest cost) path from `origin` to any node in `destinations`.

    Args:
        graph        (dict): Adjacency list.
                             { node_id (int): [(neighbour_id (int), cost (int/float)), ...] }
                             Edges are DIRECTED.
        coordinates  (dict): Node 2D positions.
                             { node_id (int): (x (int), y (int)) }
        origin        (int): Starting node ID.
        destinations  (list): List of goal node IDs. Reaching ANY ONE is success.

    Returns:
        tuple: (goal_node, nodes_created, path)
            goal_node     (int)  : The destination node reached.
            nodes_created  (int)  : Total nodes generated across ALL IDA* iterations.
                                    Includes the origin (counted as 1 at start).
            path          (list) : Node IDs from origin to goal, inclusive.

        If no path exists:
            (None, nodes_created, [])
    """

    dest_set = set(destinations)

    # ── Early exit: origin is already a destination ───────────────────────────
    if origin in dest_set:
        return origin, 1, [origin]

    # ── Initialise bound ──────────────────────────────────────────────────────
    # Start with the tightest possible admissible bound: h(origin).
    bound = _heuristic(origin, destinations, coordinates)

    # ── Node creation counter ─────────────────────────────────────────────────
    # Wrapped in a list for mutation across recursive calls.
    # Initialised to 1 (origin is the first node created).
    nodes_created_ref = [1]

    path = [origin]
    on_path = {origin}

    # ── Outer loop: increase bound until solution found or exhausted ──────────
    while True:

        t, goal, found_path = _search(
            graph, coordinates, origin, dest_set, destinations,
            0, bound, path, on_path, nodes_created_ref
        )

        if goal is not None:
            return goal, nodes_created_ref[0], found_path

        if t == math.inf:
            # No solution exists at any f-bound
            return None, nodes_created_ref[0], []

        # Raise the bound to the minimum pruned f value
        bound = t
