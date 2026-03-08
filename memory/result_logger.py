import json
import os

class ResultLogger:

    def __init__(self, path="logs/results.json"):

        self.path = path

        os.makedirs("logs", exist_ok=True)

    def log(self, data):

        with open(self.path, "a") as f:
            json.dump(data, f)
            f.write("\n")
