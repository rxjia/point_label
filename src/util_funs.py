from copy import deepcopy

import numpy as np
import json

from helper import PATH_ROOT

def save_board_param(d):
    dout = deepcopy(d)
    for k, v in dout.items():
        if isinstance(v, np.ndarray):
            dout[k] = v.tolist()
    with open(PATH_ROOT+'/config/board_param.json', 'w') as f:
        f.write(json.dumps(dout))

def load_board_param():
    with open(PATH_ROOT+'/config/board_param.json', 'r') as f:
        d = json.loads(f.read())
    for k, v in d.items():
        if isinstance(v, list):
            d[k]=np.array(v)
    return d

