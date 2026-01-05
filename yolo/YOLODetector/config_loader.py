import json

def load_config(path="./cfg/config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
