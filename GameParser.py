import pandas as pd
import numpy as np
import json
import os

import numpy as np
import typing


class Trajectory:
    def __init__(self):
        pass


class Perspective:
    EMPTY_CODE = 0
    FOOD_CODE = 1
    HAZARD_CODE = 2
    YOU_BODY_CODE = 3
    YOU_HEAD_CODE = 4
    ENEMY_BODY_CODE = 5
    ENEMY_HEAD_CODE = 6

    def valid(self, x, y):
        return x >= 0 and x < self.maxX and y >= 0 and y < self.maxY

    def __init__(self, snakeId: str, step: typing.Dict):
        self.step = step
        self.turn = step["turn"]

        self.board = step["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.head = None
        self.health = self.you["health"]
        self.length = self.you["length"]

        self.state = np.zeros((self.maxY, self.maxX))

        codes = {
            (True, True): Perspective.YOU_HEAD_CODE,
            (True, False): Perspective.YOU_BODY_CODE,
            (False, True): Perspective.ENEMY_HEAD_CODE,
            (False, False): Perspective.ENEMY_BODY_CODE,
        }

        for snake in step["board"]["snakes"]:
            if snake["id"] == snakeId:
                self.head = snake["head"]
            for i, point in enumerate(snake["body"]):
                if self.valid(point["x"], point["y"]):
                    condition = (snake["id"] == snakeId, i == 0)
                    self.state[point["y"]][point["x"]] = codes[condition]

        for point in step["board"]["hazards"]:
            self.state[point["y"]][point["x"]] = Perspective.HAZARD_CODE

        for point in step["board"]["food"]:
            self.state[point["y"]][point["x"]] = Perspective.FOOD_CODE

    @property
    def reward(self):
        x = self.head["x"]
        y = self.head["y"]

        conditions = [
            x < 0 or x >= self.maxX or y < 0 or y >= self.maxY,
            self.health == 0,
            self.state[y][x] == Perspective.YOU_BODY_CODE,
            self.state[y][x] == Perspective.ENEMY_BODY_CODE,
            self.state[y][x] == Perspective.ENEMY_HEAD_CODE,
            self.state[y][x] == Perspective.HAZARD_CODE,
        ]

        if any(conditions):
            return -1

        return 1

    def __str__(self):
        return str(self.state[::-1])


class GameParser:

    @staticmethod
    def perspectives(self, steps):
        self.perspectives = {}

        for step in steps:
            for snake in step["snakes"]:
                if snake["id"] not in self.perspectives:
                    self.perspectives[snake["id"]] = []

    def __init__(self, file_path, output_path):
        self.input_path = file_path
        self.output_path = output_path

        self.meta = {}
        self.snakeIds = None

        self.steps = None

        self.start = None
        self.end = None

    def snakeIds(self):
        for step in self.steps:
            for snake in step["snakes"]:
                if snake["id"] not in self.snakeIds:
                    self.snakeIds.append(snake["id"])

    def parse(self):
        self.steps = []
        inp = open(self.input_path, "r").readlines()

        for i, line in enumerate(inp):
            obj = json.loads(line)

            if i == 0:
                self.start = obj
                self.meta["id"] = self.start["id"]
                self.meta["map"] = self.start["map"]
            elif i == len(inp) - 1:
                self.end = obj
                self.meta["winnerId"] = self.end["winnerId"]
                self.meta["isDraw"] = self.end["isDraw"]
            else:
                step = obj

                if step["turn"] == 0:
                    snakes = step["board"]["snakes"]
                    self.snakeIds = [snake["id"] for snake in snakes]

                self.steps.append(step)

    def to_json(self):
        with open(f"{self.output_path}/output.json", "w") as f:
            json.dump(self.steps, f)


if __name__ == "__main__":
    file_path = "./data/out.log"
    output_path = "./data"
    parser = GameParser(file_path, output_path)

    parser.parse()
    parser.to_json()
