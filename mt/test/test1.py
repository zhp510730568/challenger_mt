#! /bin/usr/python

import time

import re
from mt.test.decorator import register_model

@register_model
class Decorator(object):
    def __init__(self):
        print("init Decorator")


if __name__=="__main__":
    test = Decorator()
    print(test._name_)
