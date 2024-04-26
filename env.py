import numpy as np
import typing


class BattleSnakeEnv:
    YOU_BODY_CODE = -1
    YOU_HEAD_CODE = 4
    ENEMY_BODY_CODE = -2
    ENEMY_HEAD_CODE = -4
    HAZARD_CODE = -3
    FOOD_CODE = 2
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
        self.game = None

        self.update(data)

    @property
    def reward(self):
        x = self.head["x"]
        y = self.head["y"]

        if x < 0 or x >= self.maxX or y < 0 or y >= self.maxY:
            return -1

        if self.health == 0:
            return -1

        if self.state[y][x] == BattleSnakeEnv.ENEMY_BODY_CODE:
            return -1

        if self.state[y][x] == BattleSnakeEnv.HAZARD_CODE:
            return -1

        return 1

    def update(self, data: typing.Dict):
        if self.game is None:
            self.game = np.array([self.state])
        else:
            self.game = np.append(self.game, [self.state], axis=0)

        for p in data["you"]["body"]:
            x, y = self.clamp(p["x"], p["y"])
            self.state[y][x] = BattleSnakeEnv.YOU_BODY_CODE

        for p in data["board"]["food"]:
            x, y = self.clamp(p["x"], p["y"])
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.FOOD_CODE

        for snake in data["board"]["snakes"]:
            if snake["id"] == self.you["id"]:
                continue
            for p in snake["body"]:
                x, y = self.clamp(p["x"], p["y"])
                self.state[p["y"]][p["x"]] = BattleSnakeEnv.ENEMY_BODY_CODE

        for p in data["board"]["hazards"]:
            x, y = self.clamp(p["x"], p["y"])
            self.state[p["y"]][p["x"]] = BattleSnakeEnv.HAZARD_CODE

        print(self.game.shape)

    def __str__(self):
        return str(self.state[::-1])
