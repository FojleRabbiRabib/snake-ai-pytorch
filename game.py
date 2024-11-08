import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class SnakeGameAI:

    def __init__(self, w=640, h=480, BLOCK_SIZE=20, SPEED=40):
        self.w = w                      # game width
        self.h = h                      # game height
        self.BLOCK_SIZE = BLOCK_SIZE    # dist for one move
        self.SPEED = SPEED              # game speed

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*self.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-self.BLOCK_SIZE) //
                           self.BLOCK_SIZE)*self.BLOCK_SIZE
        y = random.randint(0, (self.h-self.BLOCK_SIZE) //
                           self.BLOCK_SIZE)*self.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            reward = -0.1
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def has_escape_path(self, start_point, max_steps=5):
        """
        Use BFS to determine if there's an escape path from the snake's head within max_steps.
        This checks if moving in the given direction will trap the snake within its own body.
        """
        queue = deque([(start_point, 0)])  # (current position, step count)
        visited = set()
        visited.add(start_point)

        while queue:
            position, steps = queue.popleft()

            # If we exceed max_steps or if we've found an escape route, exit the loop
            if steps > max_steps:
                return True
            if position.x < 0 or position.x >= self.w or position.y < 0 or position.y >= self.h:
                return True  # Found an escape

            # Check adjacent positions (left, right, up, down)
            for direction in [(self.BLOCK_SIZE, 0), (-self.BLOCK_SIZE, 0), (0, self.BLOCK_SIZE), (0, -self.BLOCK_SIZE)]:
                new_position = Point(
                    position.x + direction[0], position.y + direction[1])

                if new_position not in visited and new_position not in self.snake:
                    queue.append((new_position, steps + 1))
                    visited.add(new_position)

        return False  # No escape path found

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - self.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = Point(x, y)
