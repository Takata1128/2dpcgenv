import pygame
from pygame.locals import *

WIDTH = 400
HEIGHT = 300
BACKGROUND = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    flag = False

    game = Game()
    block2 = Block(100, 100, 100, 100)

    game.append_blocks(block2)
    while not flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = True

        screen.fill(BACKGROUND)
        game.update()
        game.draw(screen)

        pygame.display.update()
        clock.tick(60)

    pygame.quit()


class Game():
    def __init__(self):
        self.player = Player(0, 275)
        self.blocks = []

    def append_blocks(self, block):
        self.blocks.append(block)

    def update(self):
        # キー入力
        pressed_key = pygame.key.get_pressed()
        vx, vy = 0, 0
        if pressed_key[K_LEFT]:
            vx -= self.player.vx
        if pressed_key[K_RIGHT]:
            vx += self.player.vx
        if pressed_key[K_UP]:
            vy -= self.player.vy
        if pressed_key[K_DOWN]:
            vy += self.player.vy
        # 衝突判定

        def check_collision_x(vx):
            newx = self.player.rect.x + vx
            newrect = Rect(newx, self.player.rect.y,
                           self.player.w, self.player.h)
            for block in self.blocks:
                collide = newrect.colliderect(block.rect)
                if collide:
                    vx = 0
                    break
            return vx

        def check_collision_y(vy):
            newy = self.player.rect.y + vy
            newrect = Rect(self.player.rect.x, newy,
                           self.player.w, self.player.h)
            for block in self.blocks:
                collide = newrect.colliderect(block.rect)
                if collide:
                    vy = 0
                    break
            return vy

        vx = check_collision_x(vx)
        vy = check_collision_y(vy)
        self.player.rect.move_ip(vx, vy)

        for block in self.blocks:
            block.update()

    def draw(self, screen):
        self.player.draw(screen)
        for block in self.blocks:
            block.draw(screen)


class Player(object):
    def __init__(self, x, y):
        self.w = 10
        self.h = 10
        self.rect = Rect(x, y, self.w, self.h)
        self.vx = 3.0
        self.vy = 3.0

    def update(self):
        pressed_key = pygame.key.get_pressed()
        if pressed_key[K_LEFT]:
            self.rect.move_ip(-self.vx, 0)
        if pressed_key[K_RIGHT]:
            self.rect.move_ip(self.vx, 0)
        if pressed_key[K_UP]:
            self.rect.move_ip(0, -self.vy)
        if pressed_key[K_DOWN]:
            self.rect.move_ip(0, self.vy)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)


class Block(object):
    def __init__(self, x, y, w, h):
        self.rect = Rect(x, y, w, h)

    def update(self):
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect)


if __name__ == "__main__":
    main()
