from collections import namedtuple
import gym
from gym.spaces import *
import numpy as np
import cv2
ACTION_NUM = 0
COOD_NUM = 4
OBS_SHAPE = (300, 400, 3)

PLAYER_SIZE = 10
GOAL_SIZE = 20
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)

STEPS_LIMIT = 100

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

        self.observation = np.full(
            self.observation_space.shape, 0, dtype=np.float32)
        self.steps = 0

    def reset(self):
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
            self.blocks.append(new_block)

        self._draw_all()
        reward = 0
        done = False
        self.steps += 1
        if self.steps == STEPS_LIMIT:
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
        if overlap(obj, self.player):
            return False
        if overlap(obj, self.goal):
            return False
        for block in self.blocks:
            if overlap(obj, block):
                return False
        return True

    def render(self):
        cv2.imshow('image', self.observation)
        # pass
