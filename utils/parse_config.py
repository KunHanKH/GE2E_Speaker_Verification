import json
import os
import sys
if os.path.join(os.path.dirname(__file__),'..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
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

