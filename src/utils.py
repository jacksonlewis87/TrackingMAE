import json


def load_json(path: str):
    with open(path, "r") as f:
        json_file = json.load(fp=f)

    return json_file


def write_json(path: str, json_object):
    with open(path, "w") as f:
        json.dump(json_object, fp=f)
