import numpy as np
import typing
import os
import subprocess


class BattleSnakeEnv:
    YOU_CODE = 1
    FOOD_CODE = 2
    ENEMY_CODE = -1
    HAZARD_CODE = -2
    EMPTY_CODE = 0

    @staticmethod
    def reset(args: typing.List[str] = None):
        print("Resetting server...")
        args = [
            "./battlesnake",
            "play",
            "--name",
            "Python Starter Project",
            "--url",
            "http://localhost:8000",
            "-g",
            "solo",
            "--browser",
            "--board-url",
            "http://localhost:5173/",
            "--foodSpawnChance",
            str(0),  # convert int to str
            "--minimumFood",
            str(0),  # convert int to str
        ]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        print(stdout.decode())
        print(stderr.decode())

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
        return str(self.state)
