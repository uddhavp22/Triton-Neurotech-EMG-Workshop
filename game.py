"""
game.py — Jump game with right-side EMG sidebar.
Game area: 0..750   Sidebar: 750..900
"""

import pygame, random

SCREEN_W, SCREEN_H = 900, 600
GAME_W   = 750          # game play area width
SIDEBAR_X = GAME_W      # sidebar starts here
SIDEBAR_W = SCREEN_W - GAME_W
GROUND_Y  = 440
FPS       = 60

# Palette (matches main.py)
BG       = ( 11,  15,  26)
PANEL    = ( 20,  27,  46)
PANEL2   = ( 28,  38,  62)
BORDER   = ( 42,  58,  94)
TEXT     = (232, 236, 248)
DIM      = ( 90, 106, 148)
ACCENT   = ( 77, 159, 255)
GREEN    = (  0, 210, 110)
RED      = (255,  72,  72)
YELLOW   = (255, 210,  50)
GROUND_C = ( 42, 160, 100)
GROUND_D = ( 28, 110,  68)
PLAYER_C = ( 77, 159, 255)
OBS_C    = (255,  72,  72)


class Player:
    W, H       = 34, 48
    JUMP_VEL   = -14.5
    GRAVITY    = 0.56

    def __init__(self):
        self.x = 100
        self.y = float(GROUND_Y - self.H)
        self.vy = 0.0
        self.on_ground = True
        self._flash = 0

    def jump(self):
        if self.on_ground:
            self.vy = self.JUMP_VEL
            self.on_ground = False
            self._flash = 10

    def update(self):
        self.vy += self.GRAVITY
        self.y  += self.vy
        if self.y >= GROUND_Y - self.H:
            self.y = float(GROUND_Y - self.H)
            self.vy = 0.0
            self.on_ground = True
        if self._flash > 0:
            self._flash -= 1

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.W, self.H)

    def draw(self, surf):
        r = self.rect()
        color = (140, 210, 255) if self._flash else PLAYER_C
        pygame.draw.rect(surf, color, r, border_radius=7)
        # Visor
        vr = pygame.Rect(r.x + 4, r.y + 6, r.w - 8, 16)
        pygame.draw.rect(surf, (20, 40, 80), vr, border_radius=4)
        pygame.draw.rect(surf, ACCENT, vr, 1, border_radius=4)
        # Legs (animated)
        leg_y = r.bottom - 4
        pygame.draw.rect(surf, (50, 100, 180), (r.x + 4,  leg_y, 10, 6), border_radius=3)
        pygame.draw.rect(surf, (50, 100, 180), (r.x + 20, leg_y, 10, 6), border_radius=3)


class Obstacle:
    TYPES = [(18, 50), (32, 32), (22, 64), (40, 24)]

    def __init__(self, speed: float):
        w, h = random.choice(self.TYPES)
        self.w, self.h = w, h
        self.x = float(GAME_W + 20)
        self.y = float(GROUND_Y - h)
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def draw(self, surf):
        r = self.rect()
        pygame.draw.rect(surf, OBS_C, r, border_radius=5)
        pygame.draw.rect(surf, (255, 140, 140), pygame.Rect(r.x, r.y, r.w, 8), border_radius=5)

    def off_screen(self):
        return self.x + self.w < 0


class EMGSidebar:
    """Right-side panel showing EMG activation and trigger state."""

    def __init__(self):
        self.x = SIDEBAR_X
        self.w = SIDEBAR_W
        # Vertical bar geometry
        self.bar_x  = self.x + 22
        self.bar_y  = 120
        self.bar_w  = 36
        self.bar_h  = 300

    def draw(self, surf, activation: float, threshold: float, fonts_small, fonts_tiny):
        # Panel background
        pygame.draw.rect(surf, PANEL, (self.x, 0, self.w, SCREEN_H))
        pygame.draw.line(surf, BORDER, (self.x, 0), (self.x, SCREEN_H), 1)

        # Title
        t = fonts_small.render("EMG", True, DIM)
        surf.blit(t, t.get_rect(center=(self.x + self.w // 2, 20)))

        is_active = activation >= threshold

        # Vertical bar background
        bar_r = pygame.Rect(self.bar_x, self.bar_y, self.bar_w, self.bar_h)
        pygame.draw.rect(surf, PANEL2, bar_r, border_radius=6)
        pygame.draw.rect(surf, BORDER, bar_r, 1, border_radius=6)

        # Fill (bottom-up)
        max_v = max(threshold * 2.5, activation * 1.2, 50.0)
        ratio = min(1.0, activation / max_v)
        fill_h = int(self.bar_h * ratio)
        fill_y = self.bar_y + self.bar_h - fill_h
        color = GREEN if is_active else ACCENT
        if fill_h > 4:
            pygame.draw.rect(surf, color,
                             (self.bar_x, fill_y, self.bar_w, fill_h),
                             border_radius=6)

        # Threshold marker
        if threshold > 0 and max_v > 0:
            thr_ratio = min(1.0, threshold / max_v)
            ty = self.bar_y + self.bar_h - int(self.bar_h * thr_ratio)
            pygame.draw.line(surf, YELLOW,
                             (self.bar_x - 6, ty),
                             (self.bar_x + self.bar_w + 6, ty), 2)
            lbl = fonts_tiny.render("THR", True, YELLOW)
            surf.blit(lbl, lbl.get_rect(center=(self.x + self.w // 2, ty - 10)))

        # µV readout
        val_color = GREEN if is_active else TEXT
        val = fonts_small.render(f"{activation:.0f}", True, val_color)
        surf.blit(val, val.get_rect(center=(self.x + self.w // 2, self.bar_y + self.bar_h + 20)))
        uv = fonts_tiny.render("µV", True, DIM)
        surf.blit(uv, uv.get_rect(center=(self.x + self.w // 2, self.bar_y + self.bar_h + 38)))

        # ACTIVE / REST pill
        pill_y = self.bar_y + self.bar_h + 65
        pill_c = GREEN if is_active else PANEL2
        pill_r = pygame.Rect(self.x + 10, pill_y, self.w - 20, 26)
        pygame.draw.rect(surf, pill_c, pill_r, border_radius=13)
        pygame.draw.rect(surf, BORDER, pill_r, 1, border_radius=13)
        pill_label = "FLEX!" if is_active else "rest"
        pill_tc = BG if is_active else DIM
        pl = fonts_tiny.render(pill_label, True, pill_tc)
        surf.blit(pl, pl.get_rect(center=pill_r.center))

        # Channel hint
        hint = fonts_tiny.render("C=cal", True, BORDER)
        surf.blit(hint, hint.get_rect(center=(self.x + self.w // 2, SCREEN_H - 30)))
        hint2 = fonts_tiny.render("M=menu", True, BORDER)
        surf.blit(hint2, hint2.get_rect(center=(self.x + self.w // 2, SCREEN_H - 14)))


class JumpGame:
    def __init__(self, surf: pygame.Surface, font_large, font_med, font_small):
        self.surf = surf
        self.font_large = font_large
        self.font_med   = font_med
        self.font_small = font_small
        self.font_tiny  = pygame.font.SysFont('Arial', 13)
        self.sidebar = EMGSidebar()
        self._reset()

    def _reset(self):
        self.state     = 'ready'
        self.player    = Player()
        self.obstacles : list[Obstacle] = []
        self.score     = 0
        self.high_score = getattr(self, 'high_score', 0)
        self.speed     = 5.0
        self.spawn_timer    = 0
        self.spawn_interval = 90
        self.frame     = 0
        self.stars = [(random.randint(0, GAME_W),
                       random.randint(20, GROUND_Y - 50),
                       random.random()) for _ in range(50)]

    def trigger_jump(self):
        if self.state == 'ready':
            self.state = 'playing'
        elif self.state == 'playing':
            self.player.jump()
        elif self.state == 'dead':
            self._reset()
            self.state = 'playing'

    def run_frame(self, activation: float, threshold: float) -> str:
        self.frame += 1

        if self.state == 'playing':
            self.player.update()

            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval:
                self.obstacles.append(Obstacle(self.speed))
                self.spawn_timer    = 0
                self.spawn_interval = random.randint(55, 110)

            for obs in self.obstacles:
                obs.update()
            self.obstacles = [o for o in self.obstacles if not o.off_screen()]

            p_rect = self.player.rect().inflate(-10, -8)
            for obs in self.obstacles:
                if p_rect.colliderect(obs.rect().inflate(-4, -4)):
                    self.state = 'dead'
                    if self.score > self.high_score:
                        self.high_score = self.score
                    break

            self.score += 1
            self.speed = 5.0 + self.score / 350.0

        self._draw(activation, threshold)
        return self.state

    def _draw(self, activation: float, threshold: float):
        s = self.surf
        s.fill(BG)

        # Stars
        for sx, sy, br in self.stars:
            c = int(60 + br * 100)
            pygame.draw.circle(s, (c, c, int(c * 1.3)), (sx, sy), 1)

        # Ground
        pygame.draw.rect(s, GROUND_C, (0, GROUND_Y, GAME_W, 8))
        pygame.draw.rect(s, GROUND_D, (0, GROUND_Y + 8, GAME_W, SCREEN_H - GROUND_Y))

        # Player + obstacles (clipped to game area)
        self.player.draw(s)
        for obs in self.obstacles:
            obs.draw(s)

        # Score HUD
        sc = self.font_med.render(f"{self.score}", True, TEXT)
        s.blit(sc, sc.get_rect(topright=(GAME_W - 20, 16)))
        hi = self.font_small.render(f"BEST  {self.high_score}", True, DIM)
        s.blit(hi, hi.get_rect(topright=(GAME_W - 20, 48)))

        # Speed indicator dots
        spd = min(10, int(self.speed))
        for i in range(10):
            c = ACCENT if i < spd else BORDER
            pygame.draw.circle(s, c, (20 + i * 14, 24), 4)

        # Sidebar
        self.sidebar.draw(s, activation, threshold, self.font_small, self.font_tiny)

        # Overlays
        if self.state == 'ready':
            self._overlay("FLEX TO START", "Flex your forearm to begin")
        elif self.state == 'dead':
            self._overlay("GAME OVER",
                          f"Score  {self.score}    Best  {self.high_score}\nFlex to restart")

    def _overlay(self, title: str, body: str):
        s = self.surf
        ov = pygame.Surface((GAME_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((5, 8, 18, 170))
        s.blit(ov, (0, 0))

        t = self.font_large.render(title, True, TEXT)
        s.blit(t, t.get_rect(center=(GAME_W // 2, SCREEN_H // 2 - 44)))

        for i, line in enumerate(body.split('\n')):
            sub = self.font_med.render(line, True, DIM)
            s.blit(sub, sub.get_rect(center=(GAME_W // 2, SCREEN_H // 2 + 16 + i * 34)))
