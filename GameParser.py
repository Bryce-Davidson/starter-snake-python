import numpy as np
import typing
import json


class Perspective:
    NUM_PLANES = 13

    EMPTY_PLANE = 0
    FOOD_PLANE = 1
    HAZARD_PLANE = 2

    YOU_BODY_PLANE = 3
    YOU_HEAD_PLANE = 4

    ENEMY_ONE_BODY_PLANE = 5
    ENEMY_ONE_HEAD_PLANE = 6
    ENEMY_TWO_BODY_PLANE = 7
    ENEMY_TWO_HEAD_PLANE = 8
    ENEMY_THREE_BODY_PLANE = 9
    ENEMY_THREE_HEAD_PLANE = 10
    ENEMY_FOUR_BODY_PLANE = 11
    ENEMY_FOUR_HEAD_PLANE = 12

    def __init__(self, snakeId: str, step: typing.Dict):
        self.turn = step["turn"]

        self.board = step["board"]
        self.maxY = self.board["height"]
        self.maxX = self.board["width"]

        self.id = snakeId
        self.head = None
        self.move = None
        self.health = None
        self.length = None

        self.enemy_ids = [
            snake["id"] for snake in step["board"]["snakes"] if snake["id"] != snakeId
        ]

        if len(self.enemy_ids) > 4:
            raise ValueError("Too many enemies")

        self.enemy_plane = {
            snakeId: (
                Perspective.ENEMY_ONE_BODY_PLANE + (i * 2),
                Perspective.ENEMY_ONE_HEAD_PLANE + (i * 2),
            )
            for i, snakeId in enumerate(self.enemy_ids)
        }

        print(self.enemy_plane)
        exit()

        self.state = np.zeros((Perspective.NUM_PLANES, self.maxY, self.maxX))

        for snake in step["board"]["snakes"]:
            IS_YOU = snake["id"] == snakeId
            if IS_YOU:
                self.move = snake["move"]
                self.health = snake["health"]
                self.head = snake["head"]
                self.length = snake["length"]

            for i, point in enumerate(snake["body"]):
                IS_HEAD = i == 0

        for point in step["board"]["hazards"]:
            self.state[Perspective.HAZARD_PLANE][point["y"]][point["x"]] = 1

        for point in step["board"]["food"]:
            self.state[Perspective.FOOD_PLANE][point["y"]][point["x"]] = 1

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

        self.meta = {}
        self.steps = []
        self.snakeIds = None
        self.perspectives = {}

        self.start = None
        self.end = None

        # --- Parse steps ---
        with open(self.input_path, "r") as log_file:
            json_objs = [json.loads(line) for line in log_file]

        self.start = json_objs[0]
        self.meta = {"id": self.start["id"], "map": self.start["map"]}

        self.end = json_objs[-1]
        self.meta.update(
            {"winnerId": self.end["winnerId"], "isDraw": self.end["isDraw"]}
        )

        self.steps = json_objs[1:-1]
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

            # print(f"snakeId: {snakeId}, turn: {i}")
            print(perspective)
            # print(perspective.action)
            # print(reward)
