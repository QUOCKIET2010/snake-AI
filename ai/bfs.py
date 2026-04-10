from collections import deque

class BFS:
    def get_path(self, start, goal, obstacles, grid_size):
        queue = deque([(start, [])])
        visited = set([start])
        visited_nodes = []

        while queue:
            current, path = queue.popleft()
            visited_nodes.append(current)

            if current == goal:
                return path, visited_nodes

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if neighbor not in visited and neighbor not in obstacles[:-1]:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        return [], visited_nodes