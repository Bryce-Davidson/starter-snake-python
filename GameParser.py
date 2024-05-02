import pandas as pd
import numpy as np
import typing
import json


class Perspective:
    # 11 = 3 entity planes + 2 you planes + 6 enemy planes (1 body, 1 head)
    NUM_PLANES = 11

    EMPTY_PLANE = 0
    FOOD_PLANE = 1
    HAZARD_PLANE = 2
    YOU_BODY_PLANE = 3
    YOU_HEAD_PLANE = 4

    def __init__(self, snakeId: str, enemyOrder: list, step: typing.Dict):
        self.turn = step["turn"]
        self.enemyOffset = {id: i + 2 for i, id in enumerate(enemyOrder)}

        self.board = step["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.id = snakeId
        self.head = None
        self.move = None
        self.health = None
        self.length = None

        if len(self.enemyOffset) > 3:
            raise ValueError("Too many enemies")

        self.state = np.zeros(
            (Perspective.NUM_PLANES, self.maxY, self.maxX), dtype=np.int8
        )

        for snake in step["board"]["snakes"]:
            if snake["id"] == self.id:
                self.move = snake["move"]
                self.health = snake["health"]
                self.head = snake["head"]
                self.length = snake["length"]

            for i, point in enumerate(snake["body"]):
                # start at you body plane
                plane = Perspective.YOU_BODY_PLANE
                # add enemy offset
                plane += self.enemyOffset.get(snake["id"], 0)
                # add 1 to plane if head
                plane += i == 0

                self.state[plane][point["y"]][point["x"]] = 1

        for point in step["board"]["hazards"]:
            self.state[Perspective.HAZARD_PLANE][point["y"]][point["x"]] = 1

        for point in step["board"]["food"]:
            self.state[Perspective.FOOD_PLANE][point["y"]][point["x"]] = 1

    def encode(self):
        flat = self.state.flatten()
        return np.append(flat, self.action).astype(np.int8).tobytes()

    @property
    def action(self):
        actions = {"up": 0, "right": 1, "down": 2, "left": 3}
        return actions[self.move]

    @property
    def view(self):
        view = np.zeros((self.maxY, self.maxX))
        for i, plane in enumerate(self.state):
            view = np.where(plane == 1, i, view)

        return view

    def __str__(self):
        string = ""
        for row in self.view[::-1]:
            for cell in row:
                string += str(int(cell)) + " "
            string += "\n"

        return string

    def __repr__(self):
        return self.__str__()


class Game:
    def __init__(self, file_path):
        self.input_path = file_path

        self.id = None
        self.map = None
        self.steps = []
        self.snakeIds = None
        self.perspectives = {}

        self.winnerId = None
        self.isDraw = None

        self.start = None
        self.end = None

        # --- Parse steps ---
        with open(self.input_path, "r") as log_file:
            objs = [json.loads(line) for line in log_file]

        self.start = objs[0]
        self.id = self.start["id"]
        self.map = self.start["map"]

        self.end = objs[-1]
        self.winnerId = self.end["winnerId"]
        self.isDraw = self.end["isDraw"]

        self.steps = objs[1:-1]

        self.snakeOrder = [snake["id"] for snake in self.steps[0]["board"]["snakes"]]

        # --- Create perspectives ---
        for step in self.steps:
            for snake in step["board"]["snakes"]:
                snakeId = snake["id"]
                if snakeId not in self.perspectives:
                    self.perspectives[snakeId] = []

                enemyOrder = self.snakeOrder.copy()
                enemyOrder.remove(snakeId)

                self.perspectives[snakeId].append(
                    Perspective(snakeId, enemyOrder, step)
                )

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

    for snakeId, persepctives in game.perspectives.items():
        for i, p in enumerate(persepctives):
            # state = p.state.flatten()
            # action = p.action

            # reward = 0
            # if (i + 1) == len(persepctives) - 1:
            # reward = 1 if game.meta["winnerId"] == snakeId else -1

            with open("./data/states.bin", "ab") as f:
                f.write(p.encode())

    print("Data written to file")
    arrays = []
    with open("./data/states.bin", "rb") as f:
        byte_data_read = f.read()
        print(len(byte_data_read))
        for i in range(0, len(byte_data_read), 11**3 + 1):
            arrays.append(
                np.frombuffer(byte_data_read[i : i + 11**3 + 1], dtype=np.int8)
            )

    for array in arrays:
        print(array)
