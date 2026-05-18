"""
gbfs.py — Greedy Best-First Search (GBFS)

Algorithm type : Informed search
Optimal        : No  — ignores path cost, only uses heuristic
Complete       : Yes — will find a solution if one exists (on finite graphs with visited tracking)
Time           : O(b^m) worst case, often much better due to heuristic guidance
Space          : O(b^m) — must store all generated nodes

Evaluation function:
    f(n) = h(n)
    where h(n) = Euclidean (straight-line) distance from node n to the nearest destination.
    The algorithm always expands the node that LOOKS closest to a goal,
    with no regard for how expensive the path to reach that node was.

Why Euclidean distance as heuristic?
    The nodes live on a 2D coordinate grid. Straight-line distance is the
    natural and standard heuristic for this kind of spatial graph.

Tiebreaking (required by spec NOTE 2):
    When two nodes have identical h values, expand in ASCENDING NODE ID order.
    When h AND node ID are identical (two different paths arriving at the same node),
    expand in CHRONOLOGICAL (insertion) order — the node added earlier goes first.
    This is enforced by the heap key:  (h_value, node_id, insertion_counter)
    Python's heapq is a min-heap, so smaller values of each field win.
"""

import math
import heapq


# ── Helpers ──────────────────────────────────────────────────────────────────

def _euclidean(coord1, coord2):
    """Straight-line distance between two (x, y) points."""
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def _heuristic(node_id, destinations, coordinates):
    """
    h(n) = minimum Euclidean distance from node_id to any destination.

    Using the MINIMUM over all destinations means we are always measuring
    distance to the nearest goal — consistent with the spec's objective of
    reaching ANY ONE of the destination nodes.
    """
    node_coord = coordinates[node_id]
    return min(_euclidean(node_coord, coordinates[d]) for d in destinations)


# ── Main algorithm ────────────────────────────────────────────────────────────

def gbfs(graph, coordinates, origin, destinations):
    """
    Greedy Best-First Search.

    Searches for a path from `origin` to any node in `destinations` by always
    expanding the frontier node with the lowest heuristic value h(n).

    Args:
        graph        (dict): Adjacency list.
                             { node_id (int): [(neighbour_id (int), cost (int/float)), ...] }
                             Edges are DIRECTED — (from, to) only, not bidirectional.
        coordinates  (dict): Node positions.
                             { node_id (int): (x (int), y (int)) }
        origin        (int): The starting node ID.
        destinations  (list): List of goal node IDs. Reaching ANY ONE is success.

    Returns:
        tuple: (goal_node, nodes_created, path)
            goal_node    (int)  : The destination node that was reached.
            nodes_created (int) : Total number of nodes generated (pushed onto frontier).
                                  This includes the origin node itself.
            path         (list) : Ordered list of node IDs from origin to goal.

        If no path exists:
            (None, nodes_created, [])
    """

    dest_set = set(destinations)

    # ── Early exit: origin is already a destination ───────────────────────────
    if origin in dest_set:
        return origin, 1, [origin]

    # ── Frontier (open list) ──────────────────────────────────────────────────
    # Each entry: (h_value, node_id, insertion_counter, path_so_far)
    #
    # Heap ordering:
    #   1st key  — h_value          : lower h = higher priority (greedy)
    #   2nd key  — node_id          : ascending order tiebreak (spec NOTE 2)
    #   3rd key  — insertion_counter: chronological tiebreak (spec NOTE 2)
    #
    # path_so_far is excluded from comparison because lists are not compared
    # by heapq once a definitive ordering is established by the first 3 keys.

    insertion_counter = 0  # increments each time a node is pushed

    start_h = _heuristic(origin, destinations, coordinates)
    frontier = []
    heapq.heappush(frontier, (start_h, origin, insertion_counter, [origin]))

    # ── Explored set (closed list) ────────────────────────────────────────────
    # Once a node is popped and expanded, it is marked explored.
    # We never expand a node twice — this prevents infinite loops on cyclic graphs.
    explored = set()

    # ── Node counter ──────────────────────────────────────────────────────────
    # Count every node the algorithm generates (creates in memory), not just those
    # it expands. The origin is the first node created.
    nodes_created = 1

    # ── Main search loop ──────────────────────────────────────────────────────
    while frontier:

        # Pop the node with the best (lowest) h value
        h_val, current, _, path = heapq.heappop(frontier)

        # If this node was already expanded via a previously popped path, skip it.
        # (Multiple copies of the same node can exist in the frontier with
        #  different paths — we only process the first / best one.)
        if current in explored:
            continue

        explored.add(current)

        # ── Goal test ─────────────────────────────────────────────────────────
        # Test on expansion (pop), not on generation (push).
        # This is the standard approach for GBFS.
        if current in dest_set:
            return current, nodes_created, path

        # ── Expand: generate successors ───────────────────────────────────────
        # Sort neighbours by node ID ascending to satisfy tiebreak rule 1
        # before even considering insertion order.
        neighbours = sorted(graph.get(current, []), key=lambda edge: edge[0])

        for neighbour_id, _edge_cost in neighbours:

            # Only generate nodes not yet fully explored
            if neighbour_id not in explored:
                insertion_counter += 1
                nodes_created += 1

                neighbour_h = _heuristic(neighbour_id, destinations, coordinates)

                heapq.heappush(
                    frontier,
                    (neighbour_h, neighbour_id, insertion_counter, path + [neighbour_id])
                )

    # ── No path found ─────────────────────────────────────────────────────────
    return None, nodes_created, []