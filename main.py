# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>

import os
import sys
import time
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from datetime import datetime
from train import train_agent


def main(args):
    """ Starts different tests
    Args:
        param1(args): args
    """
    path = args.locexp
    # experiment_name = args.experiment_name
    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    dir_model = os.path.join(path, "pytorch_models")
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    print("Created model dir {} ".format(dir_model))
    with open (args.param, "r") as f:
        param = json.load(f)
    train_agent(param)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="experiments/kuka", type=str)
    arg = parser.parse_args()
    main(arg)
