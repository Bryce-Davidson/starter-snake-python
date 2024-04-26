import pandas as pd
import numpy as np
import json
import os


class GameParser:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.input = None
        self.output = None

        self.start = None
        self.end = None

    def read(self):
        with open(self.input_path, "r") as f:
            self.input = f.readlines()

    def parse(self):
        self.output = []
        for i, line in enumerate(self.input):
            obj = json.loads(line)
            if i == 0:
                self.start = obj
            elif i == len(self.input) - 1:
                self.end = obj
            else:
                self.output.append(obj)

    def to_json(self):
        with open(f"{self.output_path}/output.json", "w") as f:
            json.dump(self.output, f)

    def normalize(self):
        self.output = pd.json_normalize(self.output)

    def write(self):
        self.normalize()
        self.output.to_csv(f"{self.output_path}/output.csv", index=False)


if __name__ == "__main__":
    input_path = "./data/out.log"
    output_path = "./data"
    parser = GameParser(input_path, output_path)

    parser.read()
    parser.parse()
    parser.to_json()
    parser.write()

    # df = pd.read_csv(f"{output_path}/output.csv")
    # print(df.columns)
