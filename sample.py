import os
from multiprocessing import Process


def launch():
    print("Launching game...")
    args = [
        "./battlesnake",
        "play",
        "--output",
        "./data/out.log",
        "--timeout",
        "10000",
        "--url",
        "http://localhost:8001",
        "--url",
        "http://localhost:8001",
        "--url",
        "http://localhost:8001",
        "--url",
        "http://localhost:8001",
        # "--browser",
        "--board-url",
        "http://localhost:5173/",
        # "-g",
        # "solo",
        # "--foodSpawnChance",
        # "0",
        # "--minimumFood",
        # "0",
    ]

    # os.system(" ".join(args) + " > /dev/null 2>&1")
    os.system(" ".join(args))


def parse():
    print("Parsing game...")
    args = [
        "python",
        "GameParser.py",
    ]

    os.system(" ".join(args))


if __name__ == "__main__":
    for i in range(1):
        game = Process(target=launch)
        game.start()
        game.join()

        parser = Process(target=parse)
        parser.start()
        parser.join()
