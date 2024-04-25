import numpy as np
import typing
import json


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
            self.state[p["y"]][p["x"]] = 1

        for snake in self.board["snakes"]:
            for p in snake["body"]:
                self.state[p["y"]][p["x"]] = 1

        for p in self.board["food"]:
            self.state[p["y"]][p["x"]] = 2

    def valid_move(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= self.maxX or y >= self.maxY:
            return False
        if self.state[y][x] == 1:
            return False

        return True

    def get_moves(self):

        print(self.state[::-1])

        moves = []
        x = self.head["x"]
        y = self.head["y"]

        if self.valid_move(x + 1, y):
            moves.append("right")
        if self.valid_move(x - 1, y):
            moves.append("left")
        if self.valid_move(x, y + 1):
            moves.append("up")
        if self.valid_move(x, y - 1):
            moves.append("down")

        print(moves)

        return moves

    def __str__(self):
        return json.dumps(self.data)


def move(state: typing.Dict) -> typing.Dict:
    state = State(state)

    # Choose a direction to move in
    moves = state.get_moves()
    if len(moves) > 0:
        return {"move": moves[np.random.randint(0, len(moves))]}
    else:
        return {"move": "left"}


#  ---  DO NOT MODIFY BELOW THIS LINE  ---


def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
