import time
from snake import Snake
from food import Food

class GameLogic:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Khởi tạo lại toàn bộ thông số trò chơi"""
        self.snake = Snake((self.grid_size // 2, self.grid_size // 2))
        self.food = Food(self.grid_size)
        self.food.respawn(self.snake.body)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.current_path = []
        self.visited_nodes = []
        self.last_step_time = 0

    def step(self, ai_algorithm):
        """Tiến hành 1 bước của trò chơi dựa trên quyết định của AI"""
        if self.game_over:
            return

        start_time = time.time()
        
        # Lấy đường đi TRỰC TIẾP từ thuật toán AI
        path, visited = ai_algorithm.get_path(self.snake.head(), self.food.position, self.snake.body, self.grid_size)
        self.current_path = path
        self.visited_nodes = visited

        next_head = None
        if path and len(path) > 0:
            next_head = path[0]
        else:
            # Fallback: Nếu AI không tìm được đường, gọi hàm sinh tồn cơ bản
            next_head = self._get_fallback_move()

        self.last_step_time = time.time() - start_time

        # ---------------------------------------------------------
        # 1. KIỂM TRA VA CHẠM (Boundary & Self-Collision) ĐÃ ĐƯỢC SỬA LỖI
        # ---------------------------------------------------------
        # Kiểm tra đâm tường
        hit_wall = (not next_head) or (next_head[0] < 0 or next_head[0] >= self.grid_size or next_head[1] < 0 or next_head[1] >= self.grid_size)
        
        # Kiểm tra đâm thân
        hit_body = False
        if not hit_wall and next_head in self.snake.body:
            # LƯU Ý: Nếu đâm vào chính cái đuôi của mình thì KHÔNG chết (vì đuôi sẽ tự nhích đi)
            if next_head == self.snake.body[-1]:
                hit_body = False 
            else:
                hit_body = True

        # Xử lý khi game over
        if hit_wall or hit_body:
            # [SỬA LỖI UI]: Ép con rắn cập nhật mảng body để giao diện vẽ cảnh đầu cắn vào thân
            if hit_body:
                self.snake.body.insert(0, next_head) 

            self.game_over = True
            
            # [Q-LEARNING]: Phạt nặng (-10) khi đâm tường/thân
            if hasattr(ai_algorithm, 'update_q_value') and hasattr(ai_algorithm, 'last_state') and ai_algorithm.last_state is not None:
                ai_algorithm.update_q_value(ai_algorithm.last_state, ai_algorithm.last_action, -10.0, None)
            
            return

        # ---------------------------------------------------------
        # 2. XỬ LÝ DI CHUYỂN HỢP LỆ
        # ---------------------------------------------------------
        self.snake.move(next_head)
        self.steps += 1

        reward = -0.1  # [Q-LEARNING]: Phạt nhẹ mỗi bước đi để AI tìm đường ngắn nhất

        if next_head == self.food.position:
            self.score += 10
            reward = 10.0  # [Q-LEARNING]: Thưởng lớn khi ăn được thức ăn
            self.food.respawn(self.snake.body)
        else:
            self.snake.shrink()

        # ---------------------------------------------------------
        # 3. CẬP NHẬT TRẠNG THÁI CHO HỌC TĂNG CƯỜNG (RL)
        # ---------------------------------------------------------
        if hasattr(ai_algorithm, 'update_q_value') and hasattr(ai_algorithm, 'last_state') and ai_algorithm.last_state is not None:
            new_state = ai_algorithm.get_state(self.snake.head(), self.food.position, self.snake.body, self.grid_size)
            ai_algorithm.update_q_value(ai_algorithm.last_state, ai_algorithm.last_action, reward, new_state)

    def _get_fallback_move(self):
        """ 
        Logic fallback cơ bản: 
        Chỉ nhìn trước 1 bước (Look-ahead 1 step) để tìm ô nào xung quanh có nhiều khoảng trống nhất. 
        Giúp rắn sống sót lâu hơn khi các thuật toán tìm kiếm truyền thống không thấy đường.
        """
        head = self.snake.head()
        best_move = None
        max_free_spaces = -1

        # Các hướng đi có thể: Lên, Xuống, Phải, Trái
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = head[0] + dx, head[1] + dy
            
            # Nếu bước đi tiếp theo nằm trong map và không đâm vào thân rắn (trừ cái đuôi)
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.snake.body[:-1]:
                
                # Đếm không gian trống quanh ô tiếp theo
                free_spaces = 0
                for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nnx, nny = nx + ddx, ny + ddy
                    if 0 <= nnx < self.grid_size and 0 <= nny < self.grid_size and (nnx, nny) not in self.snake.body:
                        free_spaces += 1
                
                # Cập nhật bước đi có nhiều không gian an toàn nhất
                if free_spaces > max_free_spaces:
                    max_free_spaces = free_spaces
                    best_move = (nx, ny)
                    
        return best_move