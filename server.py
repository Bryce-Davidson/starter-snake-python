import os
import logging
import typing
import json


from env import BattleSnakeEnv
from flask import Flask
from flask import request


def start_server(handlers: typing.Dict):
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
        print("START")
        game_state = request.get_json()
        print(f"turn: {game_state["turn"]}")
        # print(json.dumps(game_state, indent=2))
        app.env = BattleSnakeEnv(game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        print("MOVE")
        game_state = request.get_json()
        print(f"turn: {game_state["turn"]}")
        # print(json.dumps(game_state["board"]["snakes"], indent=2))
        app.env.update(game_state)
        return {"move": "down"}

    @app.post("/end")
    def on_end():
        print("END\n")
        game_state = request.get_json()
        app.env.update(game_state)
        handlers["step"](app.env)
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
