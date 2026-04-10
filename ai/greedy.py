import heapq

class Greedy:
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance

    def get_path(self, start, goal, obstacles, grid_size):
        frontier = []
        # Lưu heuristic vào heapq
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        visited_nodes = []
        visited = set([start])

        while frontier:
            _, current = heapq.heappop(frontier)
            visited_nodes.append(current)

            if current == goal:
                break

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size and neighbor not in obstacles[:-1]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Ưu tiên chỉ dựa trên khoảng cách tới mục tiêu (h(n))
                        priority = self.heuristic(goal, neighbor)
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