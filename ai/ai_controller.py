from collections import deque

class SnakeAIController:
    def __init__(self, safe_mode=True):
        # Option bật/tắt chế độ an toàn (để bạn dễ dàng so sánh)
        self.safe_mode = safe_mode

    def get_next_move(self, snake_body, food_pos, grid_size, base_algo):
        """
        Lấy bước đi tiếp theo an toàn nhất cho rắn.
        :param snake_body: list các tuple [(x, y), ...]
        :param food_pos: tuple (x, y)
        :param grid_size: int
        :param base_algo: Object của thuật toán đang chọn (A*, BFS,...)
        :return: (path, visited_nodes)
        """
        head = snake_body[0]
        
        # 1. Tìm đường ngắn nhất bằng thuật toán cơ sở
        primary_path, visited = base_algo.get_path(head, food_pos, snake_body, grid_size)

        if not self.safe_mode:
            return primary_path, visited

        # 2. BƯỚC LOOK-AHEAD: Giả lập tương lai nếu đi theo đường này
        if primary_path:
            virtual_body = self._simulate_path(snake_body, primary_path, food_pos)
            
            # Kiểm tra xem từ cái đầu mới (mới ăn mồi) có tìm được đường về đuôi không
            if self._can_reach_tail(virtual_body, grid_size):
                return primary_path, visited # Đường này an toàn tuyệt đối
            
            # Nếu không tới được đuôi, đường này là CẠM BẪY -> Bỏ qua primary_path

        # 3. FALLBACK 1: Chasing Tail (Đuổi theo cái đuôi của mình để câu giờ)
        # Nếu không có đường an toàn tới thức ăn, đi theo đuôi để kéo dài sự sống
        path_to_tail = self._get_path_to_tail(snake_body, grid_size)
        if path_to_tail:
            # Chỉ trả về 1 bước đi đầu tiên của đường tới đuôi để có cơ hội 
            # tính lại đường tới thức ăn ở frame tiếp theo
            return [path_to_tail[0]], visited

        # 4. FALLBACK 2: Safe Move (Flood Fill)
        # Nếu không tới được mồi và cũng không thấy đuôi, chọn hướng sống lâu nhất
        safe_step = self._get_safe_move(snake_body, grid_size)
        if safe_step:
            return [safe_step], visited

        # Bế tắc hoàn toàn -> Chết
        return [], visited

    # ==========================================
    # CÁC HÀM HỖ TRỢ (HELPER METHODS) TỐI ƯU HÓA
    # ==========================================

    def _simulate_path(self, snake_body, path, food_pos):
        """ Giả lập cơ thể rắn sau khi đi hết con đường (path) """
        virtual_body = list(snake_body)
        for step in path:
            virtual_body.insert(0, step) # Tiến đầu lên
            if step == food_pos:
                pass # Ăn mồi -> thân dài ra, không cắt đuôi
            else:
                virtual_body.pop() # Đi bình thường -> cắt đuôi
        return virtual_body

    def _can_reach_tail(self, snake_body, grid_size):
        """ Kiểm tra xem đầu có đường đi tới đuôi không bằng BFS tối ưu """
        head = snake_body[0]
        tail = snake_body[-1]

        # Nếu đầu nằm ngay cạnh đuôi, an toàn 100%
        if abs(head[0] - tail[0]) + abs(head[1] - tail[1]) == 1:
            return True

        # Đuôi không được coi là vật cản vì khi ta đến, đuôi đã trượt đi
        obstacles = set(snake_body[:-1]) 
        
        queue = deque([head])
        visited = {head}

        while queue:
            current = queue.popleft()
            if current == tail:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if neighbor not in visited and neighbor not in obstacles:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return False

    def _get_path_to_tail(self, snake_body, grid_size):
        """ Tìm đường tới đuôi. Lưu ý: Không dùng 'path + [neighbor]' để tránh copy list O(N) """
        head = snake_body[0]
        tail = snake_body[-1]
        obstacles = set(snake_body[:-1])

        queue = deque([head])
        came_from = {head: None}

        while queue:
            current = queue.popleft()

            if current == tail:
                break

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if neighbor not in came_from and neighbor not in obstacles:
                        came_from[neighbor] = current
                        queue.append(neighbor)

        # Truy xuất ngược đường đi (Backtracking path)
        if tail not in came_from:
            return []

        path = []
        curr = tail
        while curr != head:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        return path

    def _get_safe_move(self, snake_body, grid_size):
        """ Fallback cuối cùng: Flood fill 4 hướng, chọn hướng có nhiều không gian trống nhất """
        head = snake_body[0]
        obstacles = set(snake_body[:-1]) # Đuôi sẽ di chuyển nên không tính là vật cản
        
        best_move = None
        max_space = -1

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = head[0] + dx, head[1] + dy
            neighbor = (nx, ny)

            if 0 <= nx < grid_size and 0 <= ny < grid_size and neighbor not in obstacles:
                # Giả lập đi 1 bước vào neighbor
                virtual_body = [neighbor] + snake_body[:-1]
                # Đếm không gian sống
                space = self._flood_fill(neighbor, set(virtual_body), grid_size)
                
                if space > max_space:
                    max_space = space
                    best_move = neighbor

        return best_move

    def _flood_fill(self, start, obstacles, grid_size):
        """ Lan truyền để đếm số lượng ô trống có thể đi """
        queue = deque([start])
        visited = {start}
        space_count = 0

        while queue:
            current = queue.popleft()
            space_count += 1

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if neighbor not in visited and neighbor not in obstacles:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return space_count