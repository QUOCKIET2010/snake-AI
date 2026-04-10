import heapq

class Dijkstra:
    def get_path(self, start, goal, obstacles, grid_size):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited_nodes = []

        while frontier:
            _, current = heapq.heappop(frontier)
            visited_nodes.append(current)

            if current == goal:
                break

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size and neighbor not in obstacles[:-1]:
                    new_cost = cost_so_far[current] + 1
                    # Chỉ quan tâm chi phí đường đi g(n), không có heuristic
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost
                        heapq.heappush(frontier, (priority, neighbor))
                        came_from[neighbor] = current

        if goal not in came_from:
            return [], visited_nodes

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path, visited_nodes