import json



class Dict2dot():
    def __init__(self, input_dict = dict()):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = Dict2dot(value)
            setattr(self, key, value)

json_file = "../config/config.json"
with open(json_file, "r") as f:
    config = json.load(f)

config_param = Dict2dot(config)

