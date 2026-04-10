import random

class Food:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.position = (0, 0)

    def respawn(self, snake_body):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in snake_body:
                self.position = pos
                break