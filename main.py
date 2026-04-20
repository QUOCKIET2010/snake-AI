import pygame
from game import GameLogic
from ai.bfs import BFS
from ai.astar import AStar
from ai.rl import DQNPlaceholder
from ai.dfs import DFS
from ai.dijkstra import Dijkstra
from ai.greedy import Greedy
from ai.qlearning import QLearning
from ui.panel import ControlPanel
from ui.components import COLORS

# Kích thước tối ưu (Tỷ lệ 70% Game - 30% Panel)
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 700
GAME_AREA_SIZE = 700

class SimulatorApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI Simulator - Professional UI/UX Edition")
        self.clock = pygame.time.Clock()

        self.grid_size = 20
        self.game = GameLogic(self.grid_size)
        
        # Panel chiếm 300px bên phải
        self.panel = ControlPanel(GAME_AREA_SIZE, 0, WINDOW_WIDTH - GAME_AREA_SIZE, WINDOW_HEIGHT)

        self.running = True
        self.paused = False
        self.speeds = [10, 20, 50, 100] # Map với dropdown index
        
        # ---> BỔ SUNG MẢNG KÍCH THƯỚC MAP <---
        self.map_sizes = [5, 10, 15, 20] # Map với size_dropdown
        
        self.algos = {
            "A*": AStar(),
            "BFS": BFS(),
            "DFS": DFS(),
            "Dijkstra": Dijkstra(),
            "Greedy Best-First": Greedy(),
            "Q-Learning (Tabular)": QLearning(),
            "RL (DQN)": DQNPlaceholder()
        }
        
        # Đồng bộ danh sách thuật toán hiển thị trên UI Dropdown
        self.panel.algo_dropdown.options = list(self.algos.keys())

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Xử lý sự kiện ưu tiên (Dropdowns trước để chống click xuyên)
            if self.panel.algo_dropdown.handle_event(event):
                self.game.current_path = [] # Reset path khi đổi thuật toán
            if self.panel.speed_dropdown.handle_event(event):
                pass

            if self.panel.size_dropdown.handle_event(event):
                new_size = self.map_sizes[self.panel.size_dropdown.selected_index]
                self.grid_size = new_size          # Cập nhật size cho main
                self.game.grid_size = new_size     # Cập nhật size cho game logic
                self.game.reset()                  # Xóa bài làm lại từ đầu
                self.paused = False
            
            # Nếu dropdown đang mở thì không xử lý click các nút bên dưới
            if self.panel.algo_dropdown.is_open or self.panel.speed_dropdown.is_open or self.panel.size_dropdown.is_open:
                continue

            if self.panel.btn_toggle_play.handle_event(event):
                if not self.game.game_over:
                    self.paused = not self.paused

            if self.panel.btn_reset.handle_event(event):
                self.game.reset()
                self.paused = False

            self.panel.tg_path.handle_event(event)
            self.panel.tg_visit.handle_event(event)
            self.panel.tg_grid.handle_event(event)

    def draw_game(self):
        self.screen.fill(COLORS["bg_game"])
        
        # ==========================================================
        # ---> SỬA ĐỔI 1: CỐ ĐỊNH KÍCH THƯỚC CELL & TÍNH TOÁN OFFSET ĐỂ CĂN GIỮA <---
        # Thay vì cell_size thay đổi theo map (zoom), ta dùng cell_size cố định của map 20x20.
        TARGET_CELL_SIZE = GAME_AREA_SIZE // 20  # Cell size lý tưởng (khoảng 35 pixel)
        
        game_pixel_w = self.grid_size * TARGET_CELL_SIZE
        game_pixel_h = self.grid_size * TARGET_CELL_SIZE
        
        # Tính toán điểm bắt đầu (Căn giữa game trong vùng 700x700)
        offset_x = (GAME_AREA_SIZE - game_pixel_w) // 2
        offset_y = (GAME_AREA_SIZE - game_pixel_h) // 2
        
        # 0. Vẽ Nền vùng Game Area (để phân biệt với vùng Offset)
        game_rect = pygame.Rect(offset_x, offset_y, game_pixel_w, game_pixel_h)
        # pygame.draw.rect(self.screen, (34, 49, 63), game_rect) # Tùy chọn: nền đậm hơn
        # ==========================================================

        # 1. Vẽ Grid (Căn giữa)
        if self.panel.tg_grid.state:
            grid_surf = pygame.Surface((game_pixel_w, game_pixel_h), pygame.SRCALPHA)
            grid_color = (255, 255, 255, 15)
            for x in range(0, game_pixel_w + 1, TARGET_CELL_SIZE):
                pygame.draw.line(grid_surf, grid_color, (x, 0), (x, game_pixel_h))
            for y in range(0, game_pixel_h + 1, TARGET_CELL_SIZE):
                pygame.draw.line(grid_surf, grid_color, (0, y), (game_pixel_w, y))
            self.screen.blit(grid_surf, (offset_x, offset_y))

        # 2. Vẽ Visited Nodes (Căn giữa)
        if self.panel.tg_visit.state and self.game.visited_nodes:
            visit_surf = pygame.Surface((TARGET_CELL_SIZE, TARGET_CELL_SIZE), pygame.SRCALPHA)
            visit_surf.fill((150, 165, 166, 40)) 
            for vx, vy in self.game.visited_nodes:
                self.screen.blit(visit_surf, (offset_x + vx * TARGET_CELL_SIZE, offset_y + vy * TARGET_CELL_SIZE))

        # 3. Vẽ Path (Căn giữa - Điểm trung tâm ô được offset)
        if self.panel.tg_path.state and self.game.current_path:
            points = [(offset_x + px * TARGET_CELL_SIZE + TARGET_CELL_SIZE//2, offset_y + py * TARGET_CELL_SIZE + TARGET_CELL_SIZE//2) for px, py in self.game.current_path]
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLORS["primary"], False, points, 4)
            if points:
                pygame.draw.circle(self.screen, COLORS["primary"], points[0], 6)

        # 4. Vẽ Thức ăn (Căn giữa)
        fx, fy = self.game.food.position
        # Cập nhật vị trí và kích thước thức ăn tương ứng cell size cố định
        food_center = (offset_x + fx * TARGET_CELL_SIZE + TARGET_CELL_SIZE//2, offset_y + fy * TARGET_CELL_SIZE + TARGET_CELL_SIZE//2)
        pygame.draw.circle(self.screen, (231, 76, 60, 100), food_center, TARGET_CELL_SIZE//1.5) # Glow
        pygame.draw.circle(self.screen, COLORS["danger"], food_center, TARGET_CELL_SIZE//2.5) # Core

        # 5. Vẽ Rắn (Giao diện mới: Đầu, Thân sọc, Đuôi bo tròn mạnh)
        # ---> SỬA ĐỔI 2: ĐỊNH NGHĨA MÀU SẮC MỚI ĐỂ PHÂN BIỆT <---
        COLOR_HEAD = (46, 204, 113)      # Xanh sáng (Đầu)
        COLOR_BODY_EVEN = (39, 174, 96) # Xanh lục lục
        COLOR_BODY_ODD = (35, 155, 86)  # Xanh lục đậm hơn (làm sọc thân)
        COLOR_TAIL = (26, 188, 156)      # Xanh mòng két (Đuôi)

        body = self.game.snake.body
        direction = self.game.snake.direction # Hướng hiện tại

        for i, (sx, sy) in enumerate(body):
            # Căn giữa tọa độ vật thể
            rect = pygame.Rect(
                offset_x + sx * TARGET_CELL_SIZE + 1,
                offset_y + sy * TARGET_CELL_SIZE + 1,
                TARGET_CELL_SIZE - 2,
                TARGET_CELL_SIZE - 2
            )
            
            if i == 0: # --- ĐẦU RẮN ---
                # Tính toán hướng mắt để chỉ hướng di chuyển
                if len(body) > 1:
                    dx, dy = body[0][0] - body[1][0], body[0][1] - body[1][1]
                else: # Trường hợp rắn mới reset
                    dx, dy = direction
                
                # Bo đầu: Xác định hướng để bo tròn mạnh hơn ở phía trước
                br_tl, br_tr, br_bl, br_br = 8, 8, 8, 8 # Mặc định
                eye_pos1, eye_pos2 = (0,0), (0,0)
                cx, cy = offset_x + sx * TARGET_CELL_SIZE + TARGET_CELL_SIZE//2, offset_y + sy * TARGET_CELL_SIZE + TARGET_CELL_SIZE//2
                eye_c = COLORS["bg_game"]
                eye_r = 3
                eye_dist = TARGET_CELL_SIZE // 4

                if dx == 1:   # Right
                    br_tr, br_br = 15, 15 # Bo tròn mạnh bên phải
                    eye_pos1, eye_pos2 = (cx + eye_dist, cy - eye_dist), (cx + eye_dist, cy + eye_dist)
                elif dx == -1: # Left
                    br_tl, br_bl = 15, 15 # Bo tròn mạnh bên trái
                    eye_pos1, eye_pos2 = (cx - eye_dist, cy - eye_dist), (cx - eye_dist, cy + eye_dist)
                elif dy == 1:  # Down
                    br_bl, br_br = 15, 15 # Bo tròn mạnh bên dưới
                    eye_pos1, eye_pos2 = (cx - eye_dist, cy + eye_dist), (cx + eye_dist, cy + eye_dist)
                elif dy == -1: # Up
                    br_tl, br_tr = 15, 15 # Bo tròn mạnh bên trên
                    eye_pos1, eye_pos2 = (cx - eye_dist, cy - eye_dist), (cx + eye_dist, cy - eye_dist)

                # Vẽ Đầu
                pygame.draw.rect(self.screen, COLOR_HEAD, rect, 
                                 border_top_left_radius=br_tl, border_top_right_radius=br_tr, 
                                 border_bottom_left_radius=br_bl, border_bottom_right_radius=br_br)
                # Vẽ Mắt
                if eye_pos1 != (0,0):
                    pygame.draw.circle(self.screen, eye_c, eye_pos1, eye_r)
                    pygame.draw.circle(self.screen, eye_c, eye_pos2, eye_r)

            elif i == len(body) - 1: # --- ĐUÔI RẮN ---
                # ---> SỬA ĐỔI 3: THAY ĐỔI HÌNH DẠNG ĐUÔI (Bo tròn cực mạnh) <---
                # Bo tròn 15 pixel cho tất cả các góc để tạo thành hình gần như hình tròn
                pygame.draw.rect(self.screen, COLOR_TAIL, rect, border_radius=15) 
                
            else: # --- THÂN RẮN ---
                # Bo tròn nhẹ (4px) cho các đốt thân để tạo sự kết nối
                # Sử dụng sọc xen kẽ để dễ theo dõi
                color = COLOR_BODY_EVEN if i % 2 == 0 else COLOR_BODY_ODD
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # 6. Overlay Game Over (Căn giữa)
        if self.game.game_over:
            # Tạo overlay chỉ đè lên vùng Game Area đã offset
            overlay = pygame.Surface((game_pixel_w, game_pixel_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200)) # Black 78% opacity
            self.screen.blit(overlay, (offset_x, offset_y))
            
            font = pygame.font.SysFont("Segoe UI", game_pixel_w//8, bold=True)
            text_surf = font.render("GAME OVER", True, COLORS["danger"])
            text_rect = text_surf.get_rect(center=(offset_x + game_pixel_w//2, offset_y + game_pixel_h//2 - game_pixel_w//30))
            self.screen.blit(text_surf, text_rect)

    def run(self):
        while self.running:
            self.handle_events()

            # Lấy tên thuật toán đang được chọn từ UI
            selected_algo_name = self.panel.algo_dropdown.options[self.panel.algo_dropdown.selected_index]

            # Logic Tự động chơi lại (Auto-Reset) cho các thuật toán Học Tăng Cường
            if self.game.game_over:
                if "RL" in selected_algo_name or "Q-Learning" in selected_algo_name:
                    self.game.reset()
                    self.paused = False

            # Xử lý logic game nếu chưa Game Over và không Pause
            elif not self.paused:
                current_algo = self.algos[selected_algo_name]
                self.game.step(current_algo)

            self.draw_game()

            # Chuẩn bị Thống kê (Stats)
            status = "GAME OVER" if self.game.game_over else ("Paused" if self.paused else "Running")
            stats = {
                "Score": self.game.score,
                "Steps": self.game.steps,
                "Length": len(self.game.snake.body),
                "Time/Step": f"{self.game.last_step_time:.4f}s",
                "Status": status
            }
            
            # Cập nhật bảng điều khiển (Panel)
            self.panel.draw(self.screen, stats)
            
            pygame.display.flip()
            
            # Điều khiển FPS dựa trên Dropdown
            current_speed = self.speeds[self.panel.speed_dropdown.selected_index]
            self.clock.tick(current_speed)

        pygame.quit()

if __name__ == "__main__":
    app = SimulatorApp()
    app.run()