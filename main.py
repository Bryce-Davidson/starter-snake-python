from state import State
import numpy as np
import typing
import json


def move(state: typing.Dict) -> typing.Dict:
    print(json.dumps(state, indent=2))
    state = State(state)
    # print(state)

    # Choose a direction to move in
    moves = state.valid_moves()
    # print(moves)

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
