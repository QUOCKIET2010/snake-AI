import pygame
from ui.components import Button, Toggle, Dropdown, COLORS

class ControlPanel:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.pad_x, self.pad_y = 20, 20
        
        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_stat_label = pygame.font.SysFont("Segoe UI", 14)
        self.font_stat_val = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_score = pygame.font.SysFont("Segoe UI", 36, bold=True)
        
        # Init Components
        # 1. State Controls
        self.btn_toggle_play = Button(x + self.pad_x, y + 40, (w - 3*self.pad_x)//2, 40, "PAUSE", COLORS["warning"])
        self.btn_reset = Button(x + w//2 + self.pad_x//2, y + 40, (w - 3*self.pad_x)//2, 40, "RESET", COLORS["danger"])
        
        # 2. Algorithm Settings
        self.algo_dropdown = Dropdown(x + self.pad_x, y + 130, w - 2*self.pad_x, 35, "ALGORITHM", 
                                      ["A*", "BFS", "DFS", "Dijkstra", "Greedy Best-First", "RL (DQN)"], 0)
        
        self.speed_dropdown = Dropdown(x + self.pad_x, y + 200, w - 2*self.pad_x, 35, "SIMULATION SPEED", 
                                       ["x1 (Slow)", "x2 (Normal)", "x5 (Fast)", "x10 (Turbo)"], 1)
        
        # 3. Visual Toggles
        start_y = 480
        self.tg_path = Toggle(x + self.pad_x, start_y, w - 2*self.pad_x, 30, "Show AI Path", True)
        self.tg_visit = Toggle(x + self.pad_x, start_y + 40, w - 2*self.pad_x, 30, "Show Visited Nodes", True)
        self.tg_grid = Toggle(x + self.pad_x, start_y + 80, w - 2*self.pad_x, 30, "Show Grid Lines", True)

    def draw(self, surface, stats):
        # Draw Panel Background
        pygame.draw.rect(surface, COLORS["bg_panel"], self.rect)
        pygame.draw.line(surface, COLORS["text_muted"], (self.rect.x, 0), (self.rect.x, self.rect.h), 1)

        # 1. CONTROLS SECTION (Chỉ vẽ text và các nút bấm)
        self._draw_text(surface, "CONTROLS", self.rect.x + self.pad_x, 15, self.font_title, COLORS["text"])
        self.btn_toggle_play.draw(surface)
        self.btn_reset.draw(surface)
        
        # Update Play/Pause button color based on state
        if stats["Status"] == "Running":
            self.btn_toggle_play.text = "PAUSE"
            self.btn_toggle_play.color = COLORS["warning"]
        elif stats["Status"] == "GAME OVER":
            self.btn_toggle_play.text = "DEAD"
            self.btn_toggle_play.color = COLORS["text_muted"]
        else:
            self.btn_toggle_play.text = "RESUME"
            self.btn_toggle_play.color = COLORS["success"]

        # 2. STATISTICS SECTION (Vẽ trước để nằm dưới cùng)
        self._draw_text(surface, "STATISTICS", self.rect.x + self.pad_x, 260, self.font_title, COLORS["text"])
        
        # Score Block (Giant)
        pygame.draw.rect(surface, COLORS["surface"], (self.rect.x + self.pad_x, 290, self.rect.w - 2*self.pad_x, 70), border_radius=8)
        self._draw_text(surface, "SCORE", self.rect.x + self.pad_x + 15, 300, self.font_stat_label, COLORS["text_muted"])
        self._draw_text(surface, str(stats["Score"]), self.rect.x + self.pad_x + 15, 315, self.font_score, COLORS["success"])

        # Other Stats
        self._draw_stat_row(surface, "Steps Taken", str(stats["Steps"]), 375)
        self._draw_stat_row(surface, "Snake Length", str(stats["Length"]), 405)
        self._draw_stat_row(surface, "Avg Time/Step", stats["Time/Step"], 435)

        # 3. VISUALIZATION SECTION
        self._draw_text(surface, "VISUALIZATION", self.rect.x + self.pad_x, 455, self.font_title, COLORS["text"])
        self.tg_path.draw(surface)
        self.tg_visit.draw(surface)
        self.tg_grid.draw(surface)

        # 4. Status Footer
        status_color = COLORS["danger"] if stats["Status"] == "GAME OVER" else (COLORS["warning"] if stats["Status"] == "Paused" else COLORS["success"])
        pygame.draw.rect(surface, status_color, (self.rect.x, self.rect.h - 30, self.rect.w, 30))
        self._draw_text(surface, f"SYSTEM STATUS: {stats['Status']}", self.rect.x + self.pad_x, self.rect.h - 25, self.font_stat_label, COLORS["bg_game"])

        # ==========================================
        # 5. DRAW DROPDOWNS LAST (Để có Z-index cao nhất)
        # ==========================================
        # Vẽ Speed trước, Algo sau. Nếu Algo xổ xuống nó sẽ đè lên Speed.
        self.speed_dropdown.draw(surface) 
        self.algo_dropdown.draw(surface)
    def _draw_text(self, surface, text, x, y, font, color):
        img = font.render(text, True, color)
        surface.blit(img, (x, y))

    def _draw_stat_row(self, surface, label, value, y):
        self._draw_text(surface, label, self.rect.x + self.pad_x, y, self.font_stat_label, COLORS["text_muted"])
        val_img = self.font_stat_val.render(value, True, COLORS["text"])
        surface.blit(val_img, (self.rect.right - self.pad_x - val_img.get_width(), y))