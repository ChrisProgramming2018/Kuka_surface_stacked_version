import os
import cv2
import torch
import numpy as np
from gym.spaces import Box
from gym import Wrapper
from collections import deque


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path





def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

