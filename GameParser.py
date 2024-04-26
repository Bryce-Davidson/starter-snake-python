import pandas as pd
import numpy as np
import json
import os


class GameParser:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.data = []

    def parse(self):
        with open(self.input_path, "r") as f:
            for line in f:
                json_object = json.loads(line)
                self.data.append(json_object)

    def write(self):
        with open(self.output_path, "w") as f:
            json.dump(self.data, f)

    def normalize(self):
        self.data = pd.json_normalize(self.data)


if __name__ == "__main__":
    input_path = "./data/out.log"
    output_path = "./data/out.json"
    parser = GameParser(input_path, output_path)

    parser.parse()
