import numpy as np
import typing
import json


class BattleSnakeEnv:
    YOU_CODE = -1
    FOOD_CODE = 2
    ENEMY_CODE = -2
    HAZARD_CODE = -3
    EMPTY_CODE = 0

    def clamp(self, x, y):
        return min(max(x, 0), self.maxX - 1), min(max(y, 0), self.maxY - 1)

    def __init__(self, data: typing.Dict):
        self.data = data
        self.turn = data["turn"]

        self.board = data["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.you = data["you"]
        self.head = self.you["head"]
        self.tail = self.you["body"][-1]

        self.health = self.you["health"]
        self.length = self.you["length"]

        self.state = np.zeros((self.maxY, self.maxX))

        for p in data["you"]["body"]:
            x, y = self.clamp(p["x"], p["y"])
            self.state[y][x] = BattleSnakeEnv.YOU_CODE

        for p in self.board["food"]:
            x, y = self.clamp(p["x"], p["y"])
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.FOOD_CODE

        for snake in self.board["snakes"]:
            if snake["id"] == self.you["id"]:
                continue
            for p in snake["body"]:
                x, y = self.clamp(p["x"], p["y"])
                self.state[p["y"]][p["x"]] = BattleSnakeEnv.ENEMY_CODE

        for p in self.board["hazards"]:
            x, y = self.clamp(p["x"], p["y"])
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.HAZARD_CODE

    @property
    def reward(self):
        x = self.head["x"]
        y = self.head["y"]

        if x < 0 or x >= self.maxX or y < 0 or y >= self.maxY:
            return -1

        if self.health == 0:
            return -1

        if self.state[y][x] == BattleSnakeEnv.ENEMY_CODE:
            return -1

        if self.state[y][x] == BattleSnakeEnv.HAZARD_CODE:
            return -1

        return 1

    def update(self, data: typing.Dict):
        self.__init__(data)

    def __str__(self):
        return str(self.state[::-1])
