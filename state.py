import numpy as np
import typing


class State:
    FOOD_CODE = 2
    HAZARD_CODE = -1
    SNAKE_CODE = -1
    YOU_CODE = -1
    EMPTY_CODE = 0

    def __init__(self, data: typing.Dict):
        self.data = data
        self.board = data["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.head = data["you"]["head"]
        self.tail = data["you"]["body"][-1]

        self.state = np.zeros((self.maxY, self.maxX))

        for p in data["you"]["body"]:
            self.state[p["y"]][p["x"]] = State.YOU_CODE

        for p in self.board["food"]:
            self.state[p["y"]][p["x"]] = State.FOOD_CODE

        for snake in self.board["snakes"]:
            for p in snake["body"]:
                self.state[p["y"]][p["x"]] = State.SNAKE_CODE

        for p in self.board["hazards"]:
            self.state[p["y"]][p["x"]] = State.HAZARD_CODE
