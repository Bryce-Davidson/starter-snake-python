import numpy as np
import typing
import json


class Perspective:
    # 11 = 3 item planes + 2 you planes + 6 enemy planes (1 body, 1 head)
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

        self.state = np.zeros((Perspective.NUM_PLANES, self.maxY, self.maxX))

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
            objs = [json.loads(line) for line in log_file]

        self.start = objs[0]
        self.meta = {"id": self.start["id"], "map": self.start["map"]}

        self.end = objs[-1]
        self.meta.update(
            {"winnerId": self.end["winnerId"], "isDraw": self.end["isDraw"]}
        )

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

    print(f"game length: {len(game)}")

    trajectories = []
    for snakeId, persepctives in game.perspectives.items():
        trajectory = []
        for i, perspective in enumerate(persepctives):

            reward = 0
            trajectory.append((perspective.state, perspective.action))

            # print(f"snakeId: {snakeId}, turn: {i}")
            # print(perspective)
            # print(perspective.action)
            # print(reward)

        trajectories.append(trajectory)

    # Store the trajectories as a numpy array
    np.save(f"{output_path}/trajectories.npy", trajectories)
