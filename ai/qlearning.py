import numpy as np
import random

class QLearning:
    def __init__(self):
        # Hyperparameters (Siêu tham số)
        self.alpha = 0.1       # Tốc độ học (Learning Rate)
        self.gamma = 0.9       # Hệ số chiết khấu (Discount Factor)
        self.epsilon = 1.0     # Tỷ lệ khám phá ban đầu
        self.epsilon_decay = 0.999 # Tốc độ giảm sự khám phá
        self.epsilon_min = 0.01

        self.q_table = {}      # Bảng Q-Table
        self.last_state = None
        self.last_action = None
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right

    def get_state(self, head, food, body, grid_size):
        """Trích xuất trạng thái môi trường thành một Tuple để làm Key cho Q-Table"""
        hx, hy = head
        fx, fy = food

        # Vị trí thức ăn tương đối
        food_dir = (
            1 if fy < hy else 0, # Thức ăn ở trên
            1 if fy > hy else 0, # Thức ăn ở dưới
            1 if fx < hx else 0, # Thức ăn bên trái
            1 if fx > hx else 0  # Thức ăn bên phải
        )

        # Cảm biến nguy hiểm (Tường hoặc thân rắn) ngay sát cạnh
        dangers = []
        for dx, dy in self.actions:
            nx, ny = hx + dx, hy + dy
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size or (nx, ny) in body:
                dangers.append(1) # Có nguy hiểm
            else:
                dangers.append(0) # An toàn

        return tuple(list(food_dir) + dangers)

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state][action]

    def update_q_value(self, state, action, reward, next_state):
        """Cập nhật giá trị Q-Table dựa trên phần thưởng nhận được"""
        current_q = self.get_q_value(state, action)
        
        # Tìm giá trị Q lớn nhất của trạng thái tiếp theo
        if next_state not in self.q_table:
             self.q_table[next_state] = {a: 0.0 for a in self.actions}
        max_next_q = max(self.q_table[next_state].values())

        # Công thức Bellman
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def get_path(self, head, food, body, grid_size):
        """Hàm này mô phỏng lại get_path để hợp với GameLogic hiện tại"""
        current_state = self.get_state(head, food, body, grid_size)

        # 1. Chọn hành động (Epsilon-Greedy)
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions) # Khám phá ngẫu nhiên
        else:
            # Khai thác: Chọn hành động có Q-value cao nhất
            if current_state not in self.q_table:
                self.q_table[current_state] = {a: 0.0 for a in self.actions}
            action = max(self.q_table[current_state], key=self.q_table[current_state].get)

        # Tính toán ô đầu tiếp theo
        next_head = (head[0] + action[0], head[1] + action[1])
        
        # Lưu lại thông tin để học ở bước sau
        self.last_state = current_state
        self.last_action = action

        # Giảm dần Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return [next_head], [] # Trả về mảng 1 phần tử là ô tiếp theo