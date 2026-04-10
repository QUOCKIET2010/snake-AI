import pygame

# Bảng màu chuẩn mực (Standard Palette)
COLORS = {
    "primary": (52, 152, 219),    # Blue
    "danger": (231, 76, 60),      # Red
    "success": (46, 204, 113),    # Green
    "warning": (241, 196, 15),    # Yellow
    "bg_panel": (30, 34, 45),     # Dark Navy
    "bg_game": (18, 18, 18),      # Very Dark Gray
    "text": (236, 240, 241),      # Off-white
    "text_muted": (149, 165, 166),# Grayish
    "surface": (44, 62, 80),      # Lighter Navy
    "surface_hover": (52, 73, 94)
}

class Button:
    def __init__(self, x, y, w, h, text, color=COLORS["primary"], hover_color=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color if hover_color else self._lighten_color(color)
        self.font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.is_hovered = False

    def _lighten_color(self, color, amount=30):
        return tuple(min(255, c + amount) for c in color)

    def draw(self, surface):
        draw_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, draw_color, self.rect, border_radius=6)
        text_surf = self.font.render(self.text, True, COLORS["text"])
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered: return True
        return False

class Toggle:
    def __init__(self, x, y, w, h, text, default_state=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.state = default_state
        self.font = pygame.font.SysFont("Segoe UI", 15)

    def draw(self, surface):
        # Text
        text_surf = self.font.render(self.text, True, COLORS["text"])
        surface.blit(text_surf, (self.rect.x, self.rect.y + (self.rect.h - text_surf.get_height()) // 2))
        
        # Switch Body
        switch_w, switch_h = 40, 20
        switch_rect = pygame.Rect(self.rect.right - switch_w, self.rect.y + (self.rect.h - switch_h)//2, switch_w, switch_h)
        bg_color = COLORS["success"] if self.state else COLORS["text_muted"]
        pygame.draw.rect(surface, bg_color, switch_rect, border_radius=10)
        
        # Switch Knob
        knob_radius = 8
        knob_x = switch_rect.right - 10 if self.state else switch_rect.left + 10
        pygame.draw.circle(surface, (255, 255, 255), (knob_x, switch_rect.centery), knob_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.state = not self.state
                return True
        return False

class Dropdown:
    def __init__(self, x, y, w, h, label, options, default_index=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.options = options
        self.selected_index = default_index
        self.is_open = False
        self.font = pygame.font.SysFont("Segoe UI", 15)
        self.label_font = pygame.font.SysFont("Segoe UI", 13, bold=True)

    def draw(self, surface):
        # Lable
        label_surf = self.label_font.render(self.label, True, COLORS["text_muted"])
        surface.blit(label_surf, (self.rect.x, self.rect.y - 18))

        # Main Box
        pygame.draw.rect(surface, COLORS["surface"], self.rect, border_radius=4)
        pygame.draw.rect(surface, COLORS["text_muted"], self.rect, 1, border_radius=4)
        
        text_surf = self.font.render(self.options[self.selected_index], True, COLORS["text"])
        surface.blit(text_surf, (self.rect.x + 10, self.rect.y + (self.rect.h - text_surf.get_height()) // 2))
        
        # Draw Arrow
        arrow_color = COLORS["text"]
        cx, cy = self.rect.right - 15, self.rect.centery
        if self.is_open:
            pygame.draw.polygon(surface, arrow_color, [(cx-5, cy+3), (cx+5, cy+3), (cx, cy-4)])
        else:
            pygame.draw.polygon(surface, arrow_color, [(cx-5, cy-3), (cx+5, cy-3), (cx, cy+4)])

        # Draw Dropdown Options (if open)
        if self.is_open:
            for i, option in enumerate(self.options):
                opt_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.h, self.rect.w, self.rect.h)
                pygame.draw.rect(surface, COLORS["surface_hover"] if i == self.selected_index else COLORS["surface"], opt_rect)
                pygame.draw.rect(surface, COLORS["text_muted"], opt_rect, 1)
                opt_text = self.font.render(option, True, COLORS["text"])
                surface.blit(opt_text, (opt_rect.x + 10, opt_rect.y + (opt_rect.h - opt_text.get_height()) // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_open:
                for i in range(len(self.options)):
                    opt_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.h, self.rect.w, self.rect.h)
                    if opt_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.is_open = False
                        return True # Trả về True nếu có sự thay đổi
                self.is_open = False # Click ra ngoài thì đóng
            else:
                if self.rect.collidepoint(event.pos):
                    self.is_open = True
        return False