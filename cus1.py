"""
cus1.py — CUS1: Iterative Deepening Depth-First Search (IDDFS)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE CHOSE IDDFS FOR CUS1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The spec requires CUS1 to be:
  (1) An UNINFORMED method — no heuristic, no coordinates, no cost awareness.
  (2) Able to FIND A PATH to the goal.

We chose IDDFS because it is strictly superior to the other two uninformed
methods already in the assignment (DFS and BFS) in the following ways:

  vs DFS:
    - DFS can get lost down infinite or very long branches before finding
      a shallow solution. IDDFS guarantees it explores shallower depths first,
      so it always finds the shortest-hop path.
    - DFS is NOT complete on graphs with cycles unless visited tracking is used.
      IDDFS is complete by design.

  vs BFS:
    - BFS is complete and finds the shallowest solution, but stores the entire
      frontier in memory: O(b^d) space where b=branching factor, d=solution depth.
    - IDDFS achieves the same level-by-level guarantee as BFS while using only
      O(b * d) space — the memory of a single DFS path — making it far more
      scalable for large graphs.

  The "cost" of IDDFS is that it re-expands nodes at every depth level.
  However, because the number of nodes at depth d grows exponentially,
  the re-expansion overhead is only a constant factor (~b/(b-1) ≈ small).
  This is the classic space-time tradeoff that makes IDDFS the recommended
  uninformed search in Russell & Norvig (AIMA, 4th ed., Section 3.4.4).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM PROPERTIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Complete : Yes — will find a solution if one exists on a finite graph.
  Optimal  : Yes, in terms of NUMBER OF HOPS (fewest edges from origin to goal).
             NOT optimal in terms of edge cost — that requires an informed search.
  Time     : O(b^d) — same asymptotic complexity as BFS.
  Space    : O(b * d) — same as DFS, far better than BFS's O(b^d).
  Informed : No — uses only graph structure, no coordinates or heuristic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Outer loop: try depth_limit = 1, 2, 3, ... up to max possible depth.
  Inner call: run a standard DFS but STOP recursing when depth_limit hits 0.
  
  At each depth level, if the DFS finds a goal → return it.
  If the DFS exhausts all nodes within the limit → increase limit and retry.
  
  The first solution found is always at the shallowest depth (fewest hops),
  because we try all depth-d paths before any depth-(d+1) paths.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIEBREAKING (spec NOTE 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rule 1 — Ascending node ID: neighbours are sorted by node ID before
            each expansion. Smaller IDs are explored first.
  Rule 2 — Chronological order: DFS explores neighbours left-to-right
            after sorting, which is naturally chronological.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE COUNTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The spec defines number_of_nodes as nodes CREATED (generated) by the search.
  IDDFS re-creates nodes on each depth pass — a node at depth 2 is genuinely
  created again during the depth=3 iteration. We count every creation across
  ALL iterations, which is the accurate total for IDDFS as a whole algorithm.
"""


# ── Recursive depth-limited DFS helper ───────────────────────────────────────

def _depth_limited_dfs(graph, current, dest_set, depth_limit, path, on_path, nodes_created_ref):
    """
    Perform a depth-limited DFS from `current`.

    Args:
        graph             (dict) : Adjacency list { node_id: [(nb, cost), ...] }
        current            (int) : Node being expanded right now.
        dest_set           (set) : Set of goal node IDs.
        depth_limit        (int) : Remaining depth budget. Stop recursing when 0.
        path              (list) : Node IDs visited so far on the current branch.
        on_path            (set) : Same nodes as path, as a set for O(1) lookup.
                                   Used for CYCLE DETECTION — prevents revisiting
                                   any ancestor on the current path.
                                   Note: different from a global visited set.
                                   IDDFS deliberately allows nodes to be visited
                                   on different branches or in different depth passes.
        nodes_created_ref (list) : Single-element list [int] used as a mutable
                                   integer counter shared across all recursive calls.
                                   (Python doesn't allow primitive mutation across
                                   stack frames without this trick or nonlocal.)

    Returns:
        (goal_node (int), path (list)) if goal found within depth limit.
        (None, [])                     if goal not found.
    """

    # ── Goal test — check BEFORE depth limit ─────────────────────────────────
    # We test the current node immediately upon visiting it.
    # This is the correct position: if we tested after the depth check,
    # we would miss goals sitting exactly AT the depth limit boundary.
    if current in dest_set:
        return current, path

    # ── Depth limit reached — do not expand further ───────────────────────────
    # Return failure for this branch; the outer loop will retry with a deeper limit.
    if depth_limit == 0:
        return None, []

    # ── Expand: generate successors ───────────────────────────────────────────
    # Sort by node ID ascending — enforces spec tiebreak rule 1.
    neighbours = sorted(graph.get(current, []), key=lambda edge: edge[0])

    for neighbour_id, _edge_cost in neighbours:

        # Cycle guard: skip nodes already on the current path.
        # This prevents infinite loops (e.g. 2->3->2->3->...) while still
        # allowing nodes to be visited on OTHER branches or in later depth passes.
        if neighbour_id in on_path:
            continue

        # Generate (create) this neighbour node
        nodes_created_ref[0] += 1

        # Recurse one level deeper, passing the updated path and on_path set
        on_path.add(neighbour_id)
        result, result_path = _depth_limited_dfs(
            graph,
            neighbour_id,
            dest_set,
            depth_limit - 1,
            path + [neighbour_id],
            on_path,
            nodes_created_ref
        )
        on_path.remove(neighbour_id)  # backtrack — undo before trying next sibling

        if result is not None:
            return result, result_path

    # All neighbours exhausted without finding goal within depth limit
    return None, []


# ── Main public function ──────────────────────────────────────────────────────

def cus1(graph, coordinates, origin, destinations):
    """
    CUS1: Iterative Deepening Depth-First Search (IDDFS).

    Uninformed search — uses only graph structure (edges).
    The `coordinates` parameter is accepted for interface compatibility
    but is intentionally not used: IDDFS is uninformed by design.

    Args:
        graph        (dict): Adjacency list.
                             { node_id (int): [(neighbour_id (int), cost (int/float)), ...] }
                             Edges are DIRECTED.
        coordinates  (dict): Node 2D positions — NOT USED (uninformed search).
        origin        (int): Starting node ID.
        destinations  (list): List of goal node IDs. Reaching ANY ONE is success.

    Returns:
        tuple: (goal_node, nodes_created, path)
            goal_node     (int)  : The destination node reached.
            nodes_created  (int)  : Total nodes generated across ALL depth iterations.
                                    Includes the origin (counted as 1 at start).
            path          (list) : Node IDs from origin to goal, inclusive.

        If no path exists:
            (None, nodes_created, [])
    """

    dest_set = set(destinations)

    # ── Early exit: origin is already a destination ───────────────────────────
    if origin in dest_set:
        return origin, 1, [origin]

    # ── Node creation counter ─────────────────────────────────────────────────
    # Wrapped in a list so _depth_limited_dfs can mutate it across stack frames.
    # Initialised to 1 because the origin node is created before the loop starts.
    nodes_created_ref = [1]

    # ── Maximum meaningful depth ──────────────────────────────────────────────
    # The longest simple path (no repeated nodes) in any graph is len(nodes) - 1.
    # Going deeper than this guarantees we would revisit a node, which our
    # cycle guard already prevents. So we cap the depth at len(graph).
    max_depth = len(graph)

    # ── Outer loop: increase depth limit until solution found or exhausted ────
    for depth_limit in range(1, max_depth + 1):

        # Fresh on_path set for each iteration — IDDFS starts over from origin
        on_path = {origin}

        goal, path = _depth_limited_dfs(
            graph,
            origin,
            dest_set,
            depth_limit,
            [origin],
            on_path,
            nodes_created_ref
        )

        if goal is not None:
            return goal, nodes_created_ref[0], path

    # ── No path found at any depth up to max ──────────────────────────────────
    return None, nodes_created_ref[0], []