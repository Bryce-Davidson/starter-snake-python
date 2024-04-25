import numpy as np
import typing


class State:
    def __init__(self, data: typing.Dict):
        self.data = data
        self.board = data["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.head = data["you"]["head"]
        self.tail = data["you"]["body"][-1]

        self.state = np.zeros((self.maxY, self.maxX))

        for p in data["you"]["body"]:
            self.state[p["y"]][p["x"]] = -1

        for p in self.board["food"]:
            self.state[p["y"]][p["x"]] = 2

        for snake in self.board["snakes"]:
            for p in snake["body"]:
                self.state[p["y"]][p["x"]] = -1

        for p in self.board["hazards"]:
            self.state[p["y"]][p["x"]] = -1

    def valid(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= self.maxX or y >= self.maxY:
            return False
        if self.state[y][x] == -1:
            return False

        return True

    def valid_moves(self):

        moves = []
        x = self.head["x"]
        y = self.head["y"]

        if self.valid(x + 1, y):
            moves.append("right")
        if self.valid(x - 1, y):
            moves.append("left")
        if self.valid(x, y + 1):
            moves.append("up")
        if self.valid(x, y - 1):
            moves.append("down")

        return moves

    def __str__(self):
        return str(self.state[::-1])
