import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

# ---------------------------------------------------------
# 1. BỘ NÃO: MẠNG NƠ-RON SÂU 
# ---------------------------------------------------------
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# ---------------------------------------------------------
# 2. ĐẶC VỤ HỌC TĂNG CƯỜNG (BẢN GỐC + TẦM NHÌN XA)
# ---------------------------------------------------------
class DQNPlaceholder:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    
        self.gamma = 0.9    
        self.memory = deque(maxlen=MAX_MEMORY)
        self.steps_without_food = 0 # Bộ đếm chống kẹt đi vòng tròn
        
        self.model = Linear_QNet(14, 256, 3) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        
        self.last_state = None
        self.last_action = None
        
        # Đổi tên file lưu để Rắn học lại 3 giác quan mới
        self.model_path = "snake_brain_far_vision.pth"
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval() 
            self.n_games = 100 
            print("==> Đã tải File Trí Nhớ! Rắn đã có tầm nhìn xa.")
        else:
            print("==> Khởi tạo não bộ. Bắt đầu học né ngõ cụt...")

    def _get_free_space(self, pt, body, grid_size, max_space):
        """Thuật toán Loang (Flood Fill): Đếm số ô an toàn có thể đi được"""
        if pt[0] < 0 or pt[0] >= grid_size or pt[1] < 0 or pt[1] >= grid_size or pt in body[:-1]:
            return 0
            
        visited = set()
        visited.add(pt)
        queue = [pt]
        count = 0
        body_set = set(body[:-1]) 

        while queue and count < max_space:
            curr = queue.pop(0)
            count += 1
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if (nx, ny) not in body_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return count

    def get_state(self, head, food, body, grid_size):
        hx, hy = head
        fx, fy = food
        
        if len(body) > 1: nx, ny = body[1]
        else: nx, ny = hx - 1, hy 
            
        dir_r, dir_l, dir_d, dir_u = hx > nx, hx < nx, hy > ny, hy < ny
        if not (dir_r or dir_l or dir_d or dir_u): dir_r = True

        pt_u = (hx, hy - 1)
        pt_d = (hx, hy + 1)
        pt_l = (hx - 1, hy)
        pt_r = (hx + 1, hy)

        def is_danger(pt):
            return pt[0] < 0 or pt[0] >= grid_size or pt[1] < 0 or pt[1] >= grid_size or pt in body[:-1]

        danger_u = is_danger(pt_u)
        danger_d = is_danger(pt_d)
        danger_l = is_danger(pt_l)
        danger_r = is_danger(pt_r)

        # XÁC ĐỊNH NGÕ CỤT BẰNG TẦM NHÌN XA
        if dir_r: pt_straight, pt_right, pt_left = pt_r, pt_d, pt_u
        elif dir_l: pt_straight, pt_right, pt_left = pt_l, pt_u, pt_d
        elif dir_u: pt_straight, pt_right, pt_left = pt_u, pt_r, pt_l
        elif dir_d: pt_straight, pt_right, pt_left = pt_d, pt_l, pt_r

        snake_len = len(body)
        # Nếu số ô trống nhỏ hơn độ dài cơ thể => Đó là ngõ cụt (Trap = 1)
        trap_straight = 1 if self._get_free_space(pt_straight, body, grid_size, snake_len) < snake_len else 0
        trap_right = 1 if self._get_free_space(pt_right, body, grid_size, snake_len) < snake_len else 0
        trap_left = 1 if self._get_free_space(pt_left, body, grid_size, snake_len) < snake_len else 0

        state = [
            # 1. Nguy hiểm ngay trước mặt (Giữ nguyên bản gốc)
            (dir_u and danger_u) or (dir_d and danger_d) or (dir_l and danger_l) or (dir_r and danger_r), 
            (dir_u and danger_r) or (dir_d and danger_l) or (dir_l and danger_u) or (dir_r and danger_d), 
            (dir_u and danger_l) or (dir_d and danger_r) or (dir_l and danger_d) or (dir_r and danger_u), 
            
            # 2. NHẬN DIỆN NGÕ CỤT (3 Giác quan mới)
            trap_straight,
            trap_right,
            trap_left,

            # 3. Hướng & Thức ăn (Giữ nguyên bản gốc)
            dir_l, dir_r, dir_u, dir_d,
            fx < hx, fx > hx, fy < hy, fy > hy
        ]
        return np.array(state, dtype=int)

    def get_path(self, head, food, body, grid_size):
        self.steps_without_food += 1
        
        # Hỗ trợ chống kẹt: Đi vòng tròn vô nghĩa quá 200 bước thì reset
        if self.steps_without_food > 200:
            return [(-1, -1)], []

        state = self.get_state(head, food, body, grid_size)
        
        # Sửa lỗi Epsilon bị âm của bản gốc
        self.epsilon = max(0, 80 - self.n_games)
        final_move = [0, 0, 0] 
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        self.last_action = final_move
        self.last_state = state

        if len(body) > 1: nx, ny = body[1]
        else: nx, ny = head[0] - 1, head[1]
        
        dir_r, dir_d, dir_l, dir_u = head[0] > nx, head[1] > ny, head[0] < nx, head[1] < ny
        if not (dir_r or dir_d or dir_l or dir_u): dir_r = True
        
        clock_wise = [(0, -1), (1, 0), (0, 1), (-1, 0)] 
        idx = 0
        if dir_u: idx = 0
        elif dir_r: idx = 1
        elif dir_d: idx = 2
        elif dir_l: idx = 3

        if final_move == [1, 0, 0]: new_dir = clock_wise[idx]             
        elif final_move == [0, 1, 0]: new_dir = clock_wise[(idx + 1) % 4] 
        else: new_dir = clock_wise[(idx - 1) % 4]                         

        next_head = (head[0] + new_dir[0], head[1] + new_dir[1])
        return [next_head], []

    # ---------------------------------------------------------
    # 3. HỆ THỐNG HUẤN LUYỆN VÀ GHI NHỚ
    # ---------------------------------------------------------
    def update_q_value(self, state, action, reward, next_state):
        if reward > 0:
            self.steps_without_food = 0

        done = (reward == -10.0) or (self.steps_without_food > 200)
        
        if done:
            reward = -10.0

        if next_state is None:
            next_state = state 
            
        self.memory.append((state, action, reward, next_state, done))
        self.train_step(state, action, reward, next_state, done)
        
        if done:
            self.n_games += 1
            self.steps_without_food = 0
            self.train_long_memory()
            self.save_model()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1: 
            state, next_state = torch.unsqueeze(state, 0), torch.unsqueeze(next_state, 0)
            action, reward = torch.unsqueeze(action, 0), torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)