"""
astar.py — A* Search (AS)

Algorithm type : Informed search
Optimal        : Yes — guaranteed to find the lowest-cost path, provided the
                 heuristic is ADMISSIBLE (never overestimates the true cost).
Complete       : Yes — will always find a solution if one exists on a finite graph.
Time           : O(b^d) worst case; heuristic quality directly reduces this in practice.
Space          : O(b^d) — must keep all generated nodes in memory.

Evaluation function:
    f(n) = g(n) + h(n)

    g(n) — exact cumulative cost of the path from origin to n so far.
    h(n) — heuristic estimate of cheapest cost from n to the nearest destination.
           We use Euclidean (straight-line) distance, which is ADMISSIBLE because
           the straight-line distance can never exceed the actual travel cost along
           edges (real edges are at least as long as the straight line between nodes).

Why admissibility matters:
    If h(n) ever overestimates, A* might discard the true optimal path too early.
    Euclidean distance on a 2D coordinate graph guarantees h(n) <= true cost,
    so A* is guaranteed to return the optimal solution.

Difference from GBFS:
    GBFS uses only h(n) — purely greedy, fast but not optimal.
    A* uses g(n) + h(n) — balances actual cost incurred with estimated cost to go,
    so it explores promising nodes WITHOUT ignoring how expensive the path was.

Tiebreaking (required by spec NOTE 2):
    When two nodes have identical f values, expand in ASCENDING NODE ID order.
    When f AND node ID are identical, expand in CHRONOLOGICAL (insertion) order.
    Heap key: (f_value, node_id, insertion_counter)
    Python heapq is a min-heap, so smaller values win at every level.
"""

import math
import heapq


# ── Helpers ───────────────────────────────────────────────────────────────────

def _euclidean(coord1, coord2):
    """Straight-line distance between two (x, y) coordinate tuples."""
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def _heuristic(node_id, destinations, coordinates):
    """
    h(n) = minimum Euclidean distance from node_id to any destination.

    Taking the minimum over ALL destinations ensures we measure distance
    to the nearest goal — matching the spec's rule that reaching ANY ONE
    destination counts as success.

    Admissibility proof:
        Edge costs in this problem are the Euclidean lengths of physical edges on
        a 2D grid. The straight-line distance between two points is always <=
        any path between them. Therefore h(n) <= true cost to reach a destination.
    """
    node_coord = coordinates[node_id]
    return min(_euclidean(node_coord, coordinates[d]) for d in destinations)


# ── Main algorithm ─────────────────────────────────────────────────────────────

def astar(graph, coordinates, origin, destinations):
    """
    A* Search.

    Finds the OPTIMAL (lowest total edge cost) path from `origin` to any node
    in `destinations` by expanding nodes in order of f(n) = g(n) + h(n).

    Args:
        graph        (dict): Adjacency list.
                             { node_id (int): [(neighbour_id (int), cost (int/float)), ...] }
                             Edges are DIRECTED — (from, to) only, not bidirectional.
        coordinates  (dict): Node 2D positions.
                             { node_id (int): (x (int), y (int)) }
        origin        (int): Starting node ID.
        destinations  (list): List of goal node IDs. Reaching ANY ONE is success.

    Returns:
        tuple: (goal_node, nodes_created, path)
            goal_node     (int)  : The destination node reached.
            nodes_created  (int)  : Total nodes generated (every heappush = +1),
                                    including the origin. This is what the spec
                                    calls "number_of_nodes".
            path          (list) : Node IDs from origin to goal, inclusive.

        If no path exists:
            (None, nodes_created, [])
    """

    dest_set = set(destinations)

    # ── Early exit: origin is already a destination ───────────────────────────
    if origin in dest_set:
        return origin, 1, [origin]

    # ── g_cost table ──────────────────────────────────────────────────────────
    # Tracks the best (lowest) known cost to reach each node discovered so far.
    # Used to avoid re-pushing a node when we already have an equal or better path.
    # Key: node_id, Value: best g(node) seen so far.
    g_cost = {origin: 0}

    # ── Frontier (open list) ──────────────────────────────────────────────────
    # Min-heap. Each entry:
    #   ( f_value, node_id, insertion_counter, g_value, path_so_far )
    #
    # Heap ordering (lower = higher priority):
    #   1st — f_value          : f(n) = g(n) + h(n), core A* criterion
    #   2nd — node_id          : ascending tiebreak (spec NOTE 2, rule 1)
    #   3rd — insertion_counter: chronological tiebreak (spec NOTE 2, rule 2)
    #
    # g_value and path_so_far are payload — not used in comparison because
    # the first three keys already produce a total ordering in all practical cases.

    insertion_counter = 0  # monotonically increasing; never decrements

    start_h = _heuristic(origin, destinations, coordinates)
    start_f = 0 + start_h  # g=0 at origin

    frontier = []
    heapq.heappush(frontier, (start_f, origin, insertion_counter, 0, [origin]))

    # ── Explored set (closed list) ────────────────────────────────────────────
    # Nodes that have been permanently expanded. Because h is admissible, the
    # first time A* pops a node it has found the optimal path to it — so we
    # never need to re-expand it.
    explored = set()

    # ── Node creation counter ─────────────────────────────────────────────────
    # The spec defines number_of_nodes as nodes CREATED (generated), not expanded.
    # We increment this every time a node is pushed onto the frontier.
    # The origin node counts as 1 (created when the search is initialised).
    nodes_created = 1

    # ── Main search loop ──────────────────────────────────────────────────────
    while frontier:

        # Pop the node with the lowest f(n) value.
        # Tiebreaks are resolved automatically by heap key ordering.
        f_val, current, _, g_val, path = heapq.heappop(frontier)

        # Lazy deletion: if this node was already expanded with an equal or
        # better g value, discard this stale frontier entry.
        # This happens because we use "re-open" rather than decrease-key.
        if current in explored:
            continue

        explored.add(current)

        # ── Goal test — performed on EXPANSION (pop), not generation (push) ──
        # Testing on pop is required for optimality: it guarantees the path
        # used to reach `current` is the best possible, because A* with an
        # admissible heuristic always pops nodes in non-decreasing f order.
        if current in dest_set:
            return current, nodes_created, path

        # ── Expand: generate successor nodes ─────────────────────────────────
        # Sort neighbours by node_id ascending to enforce spec tiebreak rule 1
        # at the generation stage (matters when multiple neighbours share equal f).
        neighbours = sorted(graph.get(current, []), key=lambda edge: edge[0])

        for neighbour_id, edge_cost in neighbours:

            # Never re-expand a node that is already in the closed list
            if neighbour_id in explored:
                continue

            tentative_g = g_val + edge_cost  # cost of reaching neighbour via current

            # Only push this neighbour if we have found a strictly better path
            # to it than any path discovered previously.
            # This prunes redundant frontier entries and keeps nodes_created
            # accurate — we only count genuinely useful node creations.
            if neighbour_id not in g_cost or tentative_g < g_cost[neighbour_id]:
                g_cost[neighbour_id] = tentative_g

                insertion_counter += 1
                nodes_created += 1

                neighbour_h = _heuristic(neighbour_id, destinations, coordinates)
                neighbour_f = tentative_g + neighbour_h

                heapq.heappush(
                    frontier,
                    (neighbour_f, neighbour_id, insertion_counter, tentative_g, path + [neighbour_id])
                )

    # ── No path found ─────────────────────────────────────────────────────────
    return None, nodes_created, []