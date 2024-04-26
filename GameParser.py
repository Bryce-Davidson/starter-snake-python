import pandas as pd
import numpy as np
import json
import os


class GameParser:

    @staticmethod
    def perspectives(self, turns):
        self.perspectives = {}

        for step in turns:
            for snake in step["snakes"]:
                if snake["id"] not in self.perspectives:
                    self.perspectives[snake["id"]] = []

                self.perspectives[snake["id"]].append(snake)

    def __init__(self, file_path, output_path):
        self.input_path = file_path
        self.output_path = output_path

        self.meta = {}
        self.snakeIds = None

        self.turns = None

        self.start = None
        self.end = None

    def snakeIds(self):
        for step in self.turns:
            for snake in step["snakes"]:
                if snake["id"] not in self.snakeIds:
                    self.snakeIds.append(snake["id"])

    def parse(self):
        self.turns = []
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
                turn = obj

                if turn["turn"] == 0:
                    snakes = turn["board"]["snakes"]
                    self.snakeIds = [snake["id"] for snake in snakes]

                self.turns.append(turn)

    def to_json(self):
        with open(f"{self.output_path}/output.json", "w") as f:
            json.dump(self.turns, f)


if __name__ == "__main__":
    file_path = "./data/out.log"
    output_path = "./data"
    parser = GameParser(file_path, output_path)

    parser.parse()
    parser.to_json()
