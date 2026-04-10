"""
game.py — Jump game logic only.
Rendering is handled by GameCanvas in main.py via QPainter.
"""
import random

# Original coordinate space (rendering scales to canvas size via QPainter transform)
GAME_W   = 750
GAME_H   = 600
GROUND_Y = 440
FPS      = 60


class Player:
    W, H     = 34, 48
    JUMP_VEL = -14.5
    GRAVITY  = 0.34


    def __init__(self):
        self.x: float = 100.0
        self.y: float = float(GROUND_Y - self.H)
        self.vy: float = 0.0
        self.on_ground: bool = True
        self._flash: int = 0

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

    @property
    def flashing(self) -> bool:
        return self._flash > 0


class Obstacle:
    TYPES = [(18, 50), (32, 32), (22, 64), (40, 24)]

    def __init__(self, speed: float):
        w, h = random.choice(self.TYPES)
        self.w: int = w
        self.h: int = h
        self.x: float = float(GAME_W + 20)
        self.y: float = float(GROUND_Y - h)
        self.speed: float = speed

    def update(self):
        self.x -= self.speed

    def off_screen(self) -> bool:
        return self.x + self.w < 0


class JumpGameLogic:
    def __init__(self):
        self.high_score: int = 0
        self._reset()

    def _reset(self):
        self.state: str = 'ready'
        self.player = Player()
        self.obstacles: list[Obstacle] = []
        self.score: int = 0
        self.speed: float = 5.0
        self.spawn_timer: int = 0
        self.spawn_interval: int = 90
        self.stars: list[tuple[int, int, float]] = [
            (random.randint(0, GAME_W),
             random.randint(20, GROUND_Y - 50),
             random.random())
            for _ in range(50)
        ]

    def trigger_jump(self):
        if self.state == 'ready':
            self.state = 'playing'
        elif self.state == 'playing':
            self.player.jump()
        elif self.state == 'dead':
            prev_hi = self.high_score
            self._reset()
            self.high_score = prev_hi
            self.state = 'playing'

    def update(self):
        if self.state != 'playing':
            return
        self.player.update()

        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.obstacles.append(Obstacle(self.speed))
            self.spawn_timer = 0
            self.spawn_interval = random.randint(55, 110)

        for obs in self.obstacles:
            obs.update()
        self.obstacles = [o for o in self.obstacles if not o.off_screen()]

        # Collision (slightly inset rects for forgiveness)
        px = int(self.player.x) + 5;  pw = self.player.W - 10
        py = int(self.player.y) + 4;  ph = self.player.H - 8
        for obs in self.obstacles:
            ox = int(obs.x) + 2;  ow = obs.w - 4
            oy = int(obs.y) + 2;  oh = obs.h - 4
            if px < ox + ow and px + pw > ox and py < oy + oh and py + ph > oy:
                self.state = 'dead'
                if self.score > self.high_score:
                    self.high_score = self.score
                break

        self.score += 1
        self.speed = 5.0 + self.score / 350.0
