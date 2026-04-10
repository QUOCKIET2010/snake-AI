class DFS:
    def get_path(self, start, goal, obstacles, grid_size):
        stack = [(start, [])]
        visited = set([start])
        visited_nodes = []

        while stack:
            current, path = stack.pop()
            visited_nodes.append(current)

            if current == goal:
                return path, visited_nodes

            valid_neighbors = []
            # Lấy tất cả các hướng có thể đi được
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if neighbor not in visited and neighbor not in obstacles[:-1]:
                        valid_neighbors.append(neighbor)
            
            # FIX LỖI: Sắp xếp các neighbor theo khoảng cách Manhattan tới thức ăn.
            # Vì Stack sử dụng cơ chế LIFO (pop phần tử cuối ra trước), 
            # ta phải dùng reverse=True để ô GẦN thức ăn nhất nằm ở CUỐI danh sách.
            valid_neighbors.sort(key=lambda n: abs(n[0] - goal[0]) + abs(n[1] - goal[1]), reverse=True)

            for neighbor in valid_neighbors:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))

        return [], visited_nodes