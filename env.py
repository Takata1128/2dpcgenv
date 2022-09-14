from collections import namedtuple
import gym
from gym.spaces import *
import numpy as np
import cv2

from queue import Queue

INF = 114514

DX = [0, 1, 0, -1]
DY = [1, 0, -1, 0]

ACTION_NUM = 0
COOD_NUM = 4
OBS_SHAPE = (50, 100, 3)

PLAYER_SIZE = 5
GOAL_SIZE = 5
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)

STEPS_LIMIT = 100

MAP_GRID_H = 10
MAP_GRID_W = 20

# MAP_GRID_H = 10
# MAP_GRID_W = 10

Vec2 = namedtuple('Vec2', ['x', 'y'])


class Object():
    def __init__(self, pos):
        self.pos: Vec2 = pos
        self.lt: Vec2 = None
        self.rb: Vec2 = None


def overlap(obj1: Object, obj2: Object):
    top = min(obj1.rb.y, obj2.rb.y)
    bottom = max(obj1.lt.y, obj2.lt.y)
    left = max(obj1.lt.x, obj2.lt.x)
    right = min(obj1.rb.x, obj2.rb.x)
    print(
        f"check: left:{left}, right:{right}\n bottom:{bottom}, top:{top}")
    print("NG" if (left < right) and (bottom < top) else "OK")
    return (left < right) and (bottom < top)


class Player(Object):
    def __init__(self, pos: Vec2):
        super().__init__(pos)
        self.lt = Vec2(self.pos.x-PLAYER_SIZE//2, self.pos.y-PLAYER_SIZE//2)
        self.rb = Vec2(self.pos.x+PLAYER_SIZE//2, self.pos.y+PLAYER_SIZE//2)

    def update(self):
        pass

    def draw(self, observation):
        cv2.circle(observation, (self.pos.x+(PLAYER_SIZE//2), self.pos.y+(
            PLAYER_SIZE//2)), PLAYER_SIZE//2, COLOR_RED, thickness=-1)


class Goal(Object):
    def __init__(self, pos: Vec2):
        super().__init__(pos)
        self.lt = Vec2(self.pos.x-GOAL_SIZE//2, self.pos.y-GOAL_SIZE//2)
        self.rb = Vec2(self.pos.x+GOAL_SIZE//2, self.pos.y+GOAL_SIZE//2)

    def update(self):
        pass

    def draw(self, observation):
        cv2.circle(
            observation, (self.pos.x+(GOAL_SIZE//2), self.pos.y+(GOAL_SIZE//2)), GOAL_SIZE//2, COLOR_GREEN, thickness=-1)


class Block(Object):
    def __init__(self, pos: Vec2, size: Vec2):
        super().__init__(pos)
        self.lt: Vec2 = self.pos
        self.rb: Vec2 = Vec2(self.pos.x + size.x, self.pos.y + size.y)

    def update(self):
        pass

    def draw(self, observation):
        cv2.rectangle(observation, self.lt, self.rb,
                      color=COLOR_BLUE, thickness=-1)


class PCGEnv(gym.Env):
    def __init__(self):
        self.action_space = Box(
            low=0.0, high=1.0, shape=(ACTION_NUM+COOD_NUM,))
        self.observation_space = Box(low=0, high=1.0, shape=OBS_SHAPE,)
        self.height = self.observation_space.shape[0]
        self.width = self.observation_space.shape[1]

        self.player = None
        self.goal = None
        self.blocks = []
        self.map = None

        self.observation = np.full(
            self.observation_space.shape, 0, dtype=np.float32)
        self.steps = 0
        self.dist_prev = INF
        self.is_placed = False

    def reset(self):
        self.player = None
        self.goal = None
        self.blocks = []
        self.map = [['.'] * MAP_GRID_W
                    for i in range(MAP_GRID_H)]

        flag = np.random.random() < 0.5
        left_size = PLAYER_SIZE if flag else GOAL_SIZE
        right_size = GOAL_SIZE if flag else PLAYER_SIZE

        left = Vec2(np.random.randint(left_size, self.width//4-left_size),
                    np.random.randint(left_size, self.height-left_size))
        right = Vec2(np.random.randint(right_size+self.width*3//4, self.width-right_size),
                     np.random.randint(right_size, self.height-right_size))
        if flag:
            self.player = Player(left)
            self.goal = Goal(right)
        else:
            self.player = Player(right)
            self.goal = Goal(left)

        grid_player = self._cood_to_grid(self.player.pos)
        grid_goal = self._cood_to_grid(self.goal.pos)

        print(*self.map, sep='\n')
        print(grid_player.x, grid_player.y)
        print(grid_goal.x, grid_goal.y)
        self.map[grid_player.y][grid_player.x] = 's'
        self.map[grid_goal.y][grid_goal.x] = 'g'
        self.dist_prev = self.dijkstra()

        self.blocks = []
        self._draw_all()

        return self.observation

    def step(self, actions):
        x = int(actions[0]*self.width)
        y = int(actions[1]*self.height)
        w = int(actions[2]*self.width)
        h = int(actions[3]*self.height)

        new_block = Block(Vec2(x, y), Vec2(w, h))
        print(f'x,y,w,h = {x},{y},{w},{h}')
        if self._check_ok(new_block):
            self.is_placed = True
            grid_lt = self._cood_to_grid(new_block.lt)
            grid_rb = self._cood_to_grid(new_block.rb)
            for i in range(grid_lt.x, grid_rb.x+1):
                for j in range(grid_lt.y, grid_rb.y+1):
                    self.map[j][i] = 'w'
            self.blocks.append(new_block)
        else:
            self.is_placed = False

        self._draw_all()
        reward = self._reward()
        done = False
        self.steps += 1
        if self.steps == STEPS_LIMIT or self.dist_prev == -1:
            done = True
        return self.observation, reward, done, {}

    def _draw_all(self):
        self.observation = np.full(
            self.observation_space.shape, 0, dtype=np.float32)
        self.player.draw(self.observation)
        self.goal.draw(self.observation)
        for block in self.blocks:
            block.draw(self.observation)

    def _check_ok(self, obj: Object):
        if overlap(obj, self.player) or self.overlap_grid(obj, self.player):
            return False
        if overlap(obj, self.goal) or self.overlap_grid(obj, self.goal):
            return False
        for block in self.blocks:
            if overlap(obj, block):
                return False
        return True

    def _cood_to_grid(self, pos: Vec2) -> Vec2:
        grid_pos_x = pos.x//(self.width//MAP_GRID_W)
        grid_pos_y = pos.y//(self.height//MAP_GRID_H)
        return Vec2(clamp(grid_pos_x, 0, MAP_GRID_W-1), clamp(grid_pos_y, 0, MAP_GRID_H-1))

    def overlap_grid(self, obj1: Object, obj2: Object):
        grid_pos_obj2 = self._cood_to_grid(obj2.pos)
        grid_lt = self._cood_to_grid(obj1.lt)
        grid_rb = self._cood_to_grid(obj1.rb)
        return grid_lt.x <= grid_pos_obj2.x <= grid_rb.x and grid_lt.y <= grid_pos_obj2.y <= grid_rb.y

    def _reward(self):
        reward = 0
        dist = self.dijkstra()
        print("Dist:", dist)
        if self.is_placed:
            reward += 0.1
        if dist == -1:
            reward = -3.0
        else:
            reward += (self.dist_prev-dist)/10
        self.dist_prev = dist
        return reward

    def dijkstra(self):
        dist = [[-1]*MAP_GRID_W for i in range(MAP_GRID_H)]
        s = self._cood_to_grid(self.player.pos)
        g = self._cood_to_grid(self.goal.pos)
        que: Queue[Vec2] = Queue()
        dist[s.y][s.x] = 0
        que.put(s)
        while not que.empty():
            v = que.get()
            for dx, dy in zip(DX, DY):
                ny = v.y + dy
                nx = v.x + dx
                if 0 <= ny < MAP_GRID_H and 0 <= nx < MAP_GRID_W and self.map[ny][nx] != 'w':
                    if dist[ny][nx] != -1:
                        continue
                    dist[ny][nx] = dist[v.y][v.x] + 1
                    que.put(Vec2(nx, ny))
        return dist[g.y][g.x]

    def render(self):
        print(*self.map, sep='\n')


def clamp(n, mini, maxi):
    return min(max(n, mini), maxi)
