from state import State
import typing


class Snake:
    def __init__(self):
        pass

    def move(self, game_state: typing.Dict):
        state = State(game_state)

        return "down"


# Start server when `python main.py` is run
if __name__ == "__main__":
    snake = Snake()

    from server import run_server

    run_server({"move": snake.move})
