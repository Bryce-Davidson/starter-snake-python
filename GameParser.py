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

    def read(self):
        with open(self.input_path, "r") as f:
            self.input = f.readlines()

    def parse(self):
        self.output = []
        for line in self.input:
            json_object = json.loads(line)
            self.output.append(json_object)

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
