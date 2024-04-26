import pandas as pd
import numpy as np
import json
import os


class GameParser:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.meta = {}
        self.snakeIds = []

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
    input_path = "./data/out.log"
    output_path = "./data"
    parser = GameParser(input_path, output_path)

    parser.parse()
    parser.to_json()
