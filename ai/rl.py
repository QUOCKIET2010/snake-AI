import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 1. MẠNG NƠ-RON NHÂN TẠO (NEURAL NETWORK)
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# 2. BỘ NÃO HỌC TĂNG CƯỜNG (RL AGENT)
class DQNPlaceholder:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Tỉ lệ khám phá (Randomness)
        self.gamma = 0.9 # Tỉ lệ chiết khấu tương lai (Discount rate)
        self.memory = deque(maxlen=100000) # Bộ nhớ kinh nghiệm
        
        # Input 11 trạng thái, Output 3 hành động (Đi thẳng, Rẽ phải, Rẽ trái)
        self.model = Linear_QNet(11, 256, 3) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.last_state = None
        self.last_action = None
        self.last_head = None
        self.last_food = None
        self.last_length = 0

    def get_state(self, head, goal, obstacles, grid_size):
        """ Trích xuất 11 trạng thái từ môi trường """
        if len(obstacles) > 1:
            neck = obstacles[1]
        else:
            neck = (head[0]-1, head[1]) # Mặc định đầu hướng sang phải

        # Hướng di chuyển hiện tại
        dir_l = head[0] < neck[0]
        dir_r = head[0] > neck[0]
        dir_u = head[1] < neck[1]
        dir_d = head[1] > neck[1]

        # 4 ô xung quanh
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        def is_collision(pt):
            if pt[0] < 0 or pt[0] >= grid_size or pt[1] < 0 or pt[1] >= grid_size:
                return True
            if pt in obstacles[:-1]:
                return True
            return False

        # Nhận diện nguy hiểm ở 3 hướng (Thẳng, Phải, Trái)
        danger_straight = (dir_r and is_collision(point_r)) or (dir_l and is_collision(point_l)) or (dir_u and is_collision(point_u)) or (dir_d and is_collision(point_d))
        danger_right = (dir_u and is_collision(point_r)) or (dir_d and is_collision(point_l)) or (dir_l and is_collision(point_u)) or (dir_r and is_collision(point_d))
        danger_left = (dir_d and is_collision(point_r)) or (dir_u and is_collision(point_l)) or (dir_r and is_collision(point_u)) or (dir_l and is_collision(point_d))

        state = [
            danger_straight, danger_right, danger_left,
            dir_l, dir_r, dir_u, dir_d,
            goal[0] < head[0],  # Thức ăn ở bên trái
            goal[0] > head[0],  # Thức ăn ở bên phải
            goal[1] < head[1],  # Thức ăn ở phía trên
            goal[1] > head[1]   # Thức ăn ở phía dưới
        ]
        return np.array(state, dtype=int)

    def train_step(self, state, action, reward, next_state, done):
        """ Huấn luyện mạng Nơ-ron (Backpropagation) """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
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
        """ Huấn luyện theo lô (Experience Replay) khi game over """
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def get_path(self, start, goal, obstacles, grid_size):
        """ Hàm chính tương tác với game.py """
        head = start
        state = self.get_state(head, goal, obstacles, grid_size)

        # HỌC TỪ BƯỚC ĐI TRƯỚC (Reward Calculation)
        if self.last_state is not None:
            if len(obstacles) < self.last_length:
                # GAME OVER (Rắn bị reset độ dài)
                self.train_step(self.last_state, self.last_action, -10, state, True)
                self.memory.append((self.last_state, self.last_action, -10, state, True))
                self.train_long_memory()
                self.n_games += 1
                self.last_state = None
                print(f"Game: {self.n_games} | Epsilon: {max(0, 80 - self.n_games)}")
            else:
                # BƯỚC ĐI BÌNH THƯỜNG
                reward = 0
                if len(obstacles) > self.last_length:
                    reward = 10 # Điểm thưởng do ăn được mồi
                else:
                    # Thưởng/phạt nhẹ dựa trên việc tới gần/ra xa mồi
                    dist_now = abs(head[0]-goal[0]) + abs(head[1]-goal[1])
                    dist_old = abs(self.last_head[0]-self.last_food[0]) + abs(self.last_head[1]-self.last_food[1])
                    reward = 1 if dist_now < dist_old else -1

                self.train_step(self.last_state, self.last_action, reward, state, False)
                self.memory.append((self.last_state, self.last_action, reward, state, False))

        # ĐƯA RA QUYẾT ĐỊNH (Exploration vs Exploitation)
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Randomness (Khám phá môi trường lúc đầu)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Dùng kiến thức đã học (AI thực thi)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # DIỄN DỊCH HÀNH ĐỘNG RA TỌA ĐỘ BƯỚC ĐI
        # Lấy hướng hiện tại của rắn
        if len(obstacles) > 1:
            neck = obstacles[1]
        else:
            neck = (head[0]-1, head[1])
        dir_r = head[0] > neck[0]
        dir_d = head[1] > neck[1]
        dir_l = head[0] < neck[0]
        dir_u = head[1] < neck[1]

        clock_wise = [(0,-1), (1,0), (0,1), (-1,0)] # UP, RIGHT, DOWN, LEFT
        idx = 0
        if dir_r: idx = 1
        elif dir_d: idx = 2
        elif dir_l: idx = 3

        if np.array_equal(final_move, [1, 0, 0]):
            new_dir = clock_wise[idx] # Đi thẳng
        elif np.array_equal(final_move, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4] # Rẽ phải
        else:
            new_dir = clock_wise[(idx - 1) % 4] # Rẽ trái

        next_head = (head[0] + new_dir[0], head[1] + new_dir[1])

        # Lưu lại trạng thái để học ở khung hình kế tiếp
        self.last_state = state
        self.last_action = final_move
        self.last_head = head
        self.last_food = goal
        self.last_length = len(obstacles)

        # Trả về 1 bước đi cho game.py
        return [next_head], [start]