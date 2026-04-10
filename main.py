import pygame
from game import GameLogic
from ai.bfs import BFS
from ai.astar import AStar
from ai.rl import DQNPlaceholder
from ai.dfs import DFS
from ai.dijkstra import Dijkstra
from ai.greedy import Greedy
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
        
        self.algos = {
            "A*": AStar(),
            "BFS": BFS(),
            "DFS": DFS(),
            "Dijkstra": Dijkstra(),
            "Greedy Best-First": Greedy(),
            "RL (DQN)": DQNPlaceholder()
        }

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Xử lý sự kiện ưu tiên (Dropdowns trước để chống click xuyên)
            if self.panel.algo_dropdown.handle_event(event):
                self.game.current_path = [] # Reset path khi đổi thuật toán
            if self.panel.speed_dropdown.handle_event(event):
                pass
            
            # Nếu dropdown đang mở thì không xử lý click các nút bên dưới
            if self.panel.algo_dropdown.is_open or self.panel.speed_dropdown.is_open:
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
        cell_size = GAME_AREA_SIZE // self.grid_size

        # 1. Vẽ Grid (Tinh tế, mờ)
        if self.panel.tg_grid.state:
            grid_surf = pygame.Surface((GAME_AREA_SIZE, GAME_AREA_SIZE), pygame.SRCALPHA)
            for x in range(0, GAME_AREA_SIZE, cell_size):
                pygame.draw.line(grid_surf, (255, 255, 255, 15), (x, 0), (x, GAME_AREA_SIZE))
            for y in range(0, GAME_AREA_SIZE, cell_size):
                pygame.draw.line(grid_surf, (255, 255, 255, 15), (0, y), (GAME_AREA_SIZE, y))
            self.screen.blit(grid_surf, (0, 0))

        # 2. Vẽ Visited Nodes (Màu xám nhạt, alpha)
        if self.panel.tg_visit.state and self.game.visited_nodes:
            visit_surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            visit_surf.fill((150, 165, 166, 40)) # Gray with transparency
            for vx, vy in self.game.visited_nodes:
                self.screen.blit(visit_surf, (vx * cell_size, vy * cell_size))

        # 3. Vẽ Path (Đường kẻ liền mạch thay vì khối vuông)
        if self.panel.tg_path.state and self.game.current_path:
            points = [(px * cell_size + cell_size//2, py * cell_size + cell_size//2) for px, py in self.game.current_path]
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLORS["primary"], False, points, 4)
            # Vẽ điểm đánh dấu bước tiếp theo
            if points:
                pygame.draw.circle(self.screen, COLORS["primary"], points[0], 6)

        # 4. Vẽ Thức ăn (Hiệu ứng phát sáng nhẹ)
        fx, fy = self.game.food.position
        pygame.draw.circle(self.screen, (231, 76, 60, 100), (fx * cell_size + cell_size//2, fy * cell_size + cell_size//2), cell_size//1.5) # Glow
        pygame.draw.circle(self.screen, COLORS["danger"], (fx * cell_size + cell_size//2, fy * cell_size + cell_size//2), cell_size//2.5) # Core

        # 5. Vẽ Rắn
        body = self.game.snake.body
        for i, (sx, sy) in enumerate(body):
            rect = pygame.Rect(sx * cell_size + 1, sy * cell_size + 1, cell_size - 2, cell_size - 2)
            
            if i == 0: # Đầu rắn
                pygame.draw.rect(self.screen, COLORS["success"], rect, border_radius=6)
                
                # Direction indicator (Đôi mắt)
                if len(body) > 1:
                    dx, dy = body[0][0] - body[1][0], body[0][1] - body[1][1]
                    cx, cy = sx * cell_size + cell_size//2, sy * cell_size + cell_size//2
                    eye_color = COLORS["bg_game"]
                    if dx == 1:   # Right
                        pygame.draw.circle(self.screen, eye_color, (cx + 4, cy - 4), 3)
                        pygame.draw.circle(self.screen, eye_color, (cx + 4, cy + 4), 3)
                    elif dx == -1: # Left
                        pygame.draw.circle(self.screen, eye_color, (cx - 4, cy - 4), 3)
                        pygame.draw.circle(self.screen, eye_color, (cx - 4, cy + 4), 3)
                    elif dy == 1:  # Down
                        pygame.draw.circle(self.screen, eye_color, (cx - 4, cy + 4), 3)
                        pygame.draw.circle(self.screen, eye_color, (cx + 4, cy + 4), 3)
                    elif dy == -1: # Up
                        pygame.draw.circle(self.screen, eye_color, (cx - 4, cy - 4), 3)
                        pygame.draw.circle(self.screen, eye_color, (cx + 4, cy - 4), 3)
            else: # Thân rắn
                pygame.draw.rect(self.screen, (39, 174, 96), rect, border_radius=4) # Màu xanh đậm hơn

        # 6. Overlay Game Over
        if self.game.game_over:
            overlay = pygame.Surface((GAME_AREA_SIZE, GAME_AREA_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)) # Black 70% opacity
            self.screen.blit(overlay, (0, 0))
            
            font = pygame.font.SysFont("Segoe UI", 60, bold=True)
            text_surf = font.render("GAME OVER", True, COLORS["danger"])
            text_rect = text_surf.get_rect(center=(GAME_AREA_SIZE//2, GAME_AREA_SIZE//2 - 20))
            self.screen.blit(text_surf, text_rect)
            
            font_sub = pygame.font.SysFont("Segoe UI", 20)
            sub_surf = font_sub.render("Press RESET to try again", True, COLORS["text"])
            self.screen.blit(sub_surf, sub_surf.get_rect(center=(GAME_AREA_SIZE//2, GAME_AREA_SIZE//2 + 30)))

    def run(self):
        while self.running:
            self.handle_events()

            if not self.paused and not self.game.game_over:
                selected_algo_name = self.panel.algo_dropdown.options[self.panel.algo_dropdown.selected_index]
                current_algo = self.algos[selected_algo_name]
                self.game.step(current_algo)

            self.draw_game()

            # Chuẩn bị Stats
            status = "GAME OVER" if self.game.game_over else ("Paused" if self.paused else "Running")
            stats = {
                "Score": self.game.score,
                "Steps": self.game.steps,
                "Length": len(self.game.snake.body),
                "Time/Step": f"{self.game.last_step_time:.4f}s",
                "Status": status
            }
            
            # Cập nhật Panel
            self.panel.draw(self.screen, stats)
            
            pygame.display.flip()
            
            # Điều khiển FPS dựa trên Dropdown
            current_speed = self.speeds[self.panel.speed_dropdown.selected_index]
            self.clock.tick(current_speed)

        pygame.quit()

if __name__ == "__main__":
    app = SimulatorApp()
    app.run()