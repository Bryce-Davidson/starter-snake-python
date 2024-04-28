import pandas as pd
import numpy as np
import json
import os

import numpy as np
import typing


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
        self.move = None
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
                self.head = np.array([snake["head"]["x"], snake["head"]["y"]])
                self.health = snake["health"]
                self.length = snake["length"]
                self.move = snake["move"]

            for i, point in enumerate(snake["body"]):
                IS_HEAD = i == 0
                condition = (snake["id"] == snakeId, IS_HEAD)
                self.state[point["y"]][point["x"]] = codes[condition]

        for point in step["board"]["hazards"]:
            self.state[point["y"]][point["x"]] = Perspective.HAZARD_CODE

        for point in step["board"]["food"]:
            self.state[point["y"]][point["x"]] = Perspective.FOOD_CODE

    @property
    def action(self):
        actions = {"up": 0, "right": 1, "down": 2, "left": 3}
        return actions[self.move]

    def __str__(self):
        string = ""
        for row in self.state[::-1]:
            for cell in row:
                string += str(int(cell)) + " "
            string += "\n"

        return string

    def __repr__(self):
        return self.__str__()


class Game:
    def __init__(self, file_path):
        self.input_path = file_path

        self.meta = {}
        self.steps = []
        self.snakeIds = None
        self.perspectives = {}

        self.start = None
        self.end = None

        # --- Parse steps ---
        with open(self.input_path, "r") as log_file:
            log = [json.loads(line) for line in log_file]

        self.start = log[0]
        self.meta = {"id": self.start["id"], "map": self.start["map"]}

        self.end = log[-1]
        self.meta.update(
            {"winnerId": self.end["winnerId"], "isDraw": self.end["isDraw"]}
        )

        self.steps = log[1:-1]
        self.snakeIds = [snake["id"] for snake in self.steps[0]["board"]["snakes"]]

        # --- Create perspectives ---
        for step in self.steps:
            for snake in step["board"]["snakes"]:
                snakeId = snake["id"]
                if snakeId not in self.perspectives:
                    self.perspectives[snakeId] = []
                self.perspectives[snakeId].append(Perspective(snakeId, step))

    def __len__(self):
        return len(self.steps)

    def to_json(self, output_path):
        with open(f"{output_path}/output.json", "w") as f:
            json.dump(self.steps, f)


if __name__ == "__main__":
    file_path = "./data/out.log"
    output_path = "./data"

    game = Game(file_path)
    game.to_json(output_path)

    print(f"game length: {len(game)}")

    for snakeId, persepctives in game.perspectives.items():
        trajectory = []
        for i, perspective in enumerate(persepctives):

            reward = 0
            trajectory.append((perspective.state, perspective.action))

            print(f"snakeId: {snakeId}, turn: {i}")
            print(perspective)
            print(perspective.action)
            print(reward)
