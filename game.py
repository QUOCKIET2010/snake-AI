import time
from snake import Snake
from food import Food

class GameLogic:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = Snake((self.grid_size//2, self.grid_size//2))
        self.food = Food(self.grid_size)
        self.food.respawn(self.snake.body)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.current_path = []
        self.visited_nodes = []
        self.last_step_time = 0

    def step(self, ai_algorithm):
        if self.game_over:
            return

        start_time = time.time()
        
        # Lấy đường đi TRỰC TIẾP từ thuật toán AI (không qua Controller trung gian)
        path, visited = ai_algorithm.get_path(self.snake.head(), self.food.position, self.snake.body, self.grid_size)
        self.current_path = path
        self.visited_nodes = visited

        next_head = None
        if path and len(path) > 0:
            next_head = path[0]
        else:
            # Fallback cũ: Nếu không thấy đường tới thức ăn, gọi hàm sinh tồn cơ bản
            next_head = self._get_fallback_move()

        self.last_step_time = time.time() - start_time

        if not next_head:
            self.game_over = True
            return

        self.snake.move(next_head)
        self.steps += 1

        if next_head == self.food.position:
            self.score += 10
            self.food.respawn(self.snake.body)
        else:
            self.snake.shrink()

    def _get_fallback_move(self):
        """ 
        Logic fallback cơ bản: 
        Chỉ nhìn trước 1 bước (Look-ahead 1 step) để tìm ô nào xung quanh có nhiều khoảng trống nhất. 
        Bạn có thể thay đổi, phát triển hoặc tối ưu hàm này theo ý thích!
        """
        head = self.snake.head()
        best_move = None
        max_free_spaces = -1

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = head[0] + dx, head[1] + dy
            # Nếu bước đi tiếp theo nằm trong map và không đâm vào thân rắn
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.snake.body[:-1]:
                
                # Đếm không gian trống quanh ô tiếp theo
                free_spaces = 0
                for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nnx, nny = nx + ddx, ny + ddy
                    if 0 <= nnx < self.grid_size and 0 <= nny < self.grid_size and (nnx, nny) not in self.snake.body:
                        free_spaces += 1
                
                # Cập nhật bước đi tối ưu nhất (có nhiều không gian nhất)
                if free_spaces > max_free_spaces:
                    max_free_spaces = free_spaces
                    best_move = (nx, ny)
                    
        return best_move