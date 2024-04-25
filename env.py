import numpy as np
import typing


class BattleSnakeEnv:
    YOU_CODE = 1
    FOOD_CODE = 2
    ENEMY_CODE = -1
    HAZARD_CODE = -2
    EMPTY_CODE = 0

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
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.YOU_CODE

        for p in self.board["food"]:
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.FOOD_CODE

        for snake in self.board["snakes"]:
            for p in snake["body"]:
                self.state[p["y"]][p["x"]] = BattleSnakeEnv.ENEMY_CODE

        for p in self.board["hazards"]:
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.HAZARD_CODE

    @classmethod
    def reset(cls):
        pass

    @property
    def reward(self):
        x = self.head["x"]
        y = self.head["y"]

        if self.health == 0:
            return -1

        if self.state[y][x] == BattleSnakeEnv.ENEMY_CODE:
            return -1

        if self.state[y][x] == BattleSnakeEnv.HAZARD_CODE:
            return -1

        if x < 0 or x >= self.maxX or y < 0 or y >= self.maxY:
            return -1

        return 1

    def update(self, data: typing.Dict):
        self.__init__(data)

    def step(self):
        return self.state, self.reward
