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

    def __init__(self, snakeId: str, step: typing.Dict):
        self.turn = step["turn"]

        self.board = step["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.head = None
        self.health = None
        self.length = None

        self.state = np.zeros((self.maxY, self.maxX))

        codes = {
            (True, True): Perspective.YOU_HEAD_CODE,
            (True, False): Perspective.YOU_BODY_CODE,
            (False, True): Perspective.ENEMY_HEAD_CODE,
            (False, False): Perspective.ENEMY_BODY_CODE,
        }

        for snake in step["board"]["snakes"]:
            if snake["id"] == snakeId:
                self.head = np.array([snake["head"]["y"], snake["head"]["x"]])
                self.health = snake["health"]
                self.length = snake["length"]

            for i, point in enumerate(snake["body"]):
                condition = (snake["id"] == snakeId, i == 0)
                self.state[point["y"]][point["x"]] = codes[condition]

        for point in step["board"]["hazards"]:
            self.state[point["y"]][point["x"]] = Perspective.HAZARD_CODE

        for point in step["board"]["food"]:
            self.state[point["y"]][point["x"]] = Perspective.FOOD_CODE

    def __str__(self):
        string = ""
        for row in self.state:
            for cell in row:
                string += str(int(cell)) + " "
            string += "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def __sub__(self, prev):
        actions = [
            np.array([0, 1]),  # right
            np.array([0, -1]),  # left
            np.array([1, 0]),  # down
            np.array([-1, 0]),  # up
        ]

        return actions.index(self.head - prev.head)


class GameParser:

    def __init__(self, file_path, output_path):
        self.input_path = file_path
        self.output_path = output_path

        self.meta = None
        self.snakeIds = None

        self.steps = None
        self.perspectives = None

        self.start = None
        self.end = None

    def parse(self):
        self.meta = {}
        self.steps = []

        # --- Parse steps ---

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

        # --- Create perspectives ---

        self.perspectives = {}

        for i, step in enumerate(self.steps):
            for snake in step["board"]["snakes"]:
                snakeId = snake["id"]

                if snakeId not in self.perspectives:
                    self.perspectives[snakeId] = []

                self.perspectives[snakeId].append(Perspective(snakeId, step))

    def to_json(self):
        with open(f"{self.output_path}/output.json", "w") as f:
            json.dump(self.steps, f)


if __name__ == "__main__":
    file_path = "./data/out.log"
    output_path = "./data"
    parser = GameParser(file_path, output_path).parse()

    parser.to_json()

    for snakeId, trajectory in parser.perspectives.values():
        for perspective in trajectory:
            print(perspective.turn)
            print(perspective)
