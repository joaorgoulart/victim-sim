import heapq
from vs.constants import VS

class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(position, map_obj):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        new_pos = (position[0] + dx, position[1] + dy)
        if map_obj.in_map(new_pos):
            actions_results = map_obj.get_actions_results(new_pos)
            if actions_results[0] != VS.WALL and actions_results[0] != VS.END:
                neighbors.append(new_pos)
    return neighbors

def a_star(start, goal, map_obj, max_iterations=10000):
    start_node = Node(start, 0, manhattan_distance(start, goal))
    open_list = [start_node]
    closed_set = set()
    iterations = 0

    while open_list and iterations < max_iterations:
        iterations += 1
        current_node = heapq.heappop(open_list)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)

        for neighbor_pos in get_neighbors(current_node.position, map_obj):
            if neighbor_pos in closed_set:
                continue

            g_cost = current_node.g_cost + 1
            h_cost = manhattan_distance(neighbor_pos, goal)
            neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)

            if neighbor_node not in open_list:
                heapq.heappush(open_list, neighbor_node)
            else:
                existing_node = next(node for node in open_list if node.position == neighbor_pos)
                if g_cost < existing_node.g_cost:
                    existing_node.g_cost = g_cost
                    existing_node.f_cost = g_cost + existing_node.h_cost
                    existing_node.parent = current_node
                    heapq.heapify(open_list)

    print(f"A* search failed after {iterations} iterations")
    return None  # No path found or max iterations reached