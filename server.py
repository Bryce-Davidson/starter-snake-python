import os
import logging
import typing

from env import BattleSnakeEnv
from flask import Flask
from flask import request


def run_server(handlers: typing.Dict):
    app = Flask("Battlesnake")
    app.env = None

    @app.get("/")
    def on_info():
        print("INFO")
        return {
            "apiversion": "1",
            "author": "",  # TODO: Your Battlesnake Username
            "color": "#888888",  # TODO: Choose color
            "head": "default",  # TODO: Choose head
            "tail": "default",  # TODO: Choose tail
        }

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        app.env = BattleSnakeEnv(game_state)
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        app.env.update(game_state)

        return handlers["move"](app.env)

    @app.post("/end")
    def on_end():
        print("GAME OVER\n")
        return "ok"

    @app.after_request
    def identify_server(response):
        response.headers.set("server", "battlesnake/github/starter-snake-python")
        return response

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
