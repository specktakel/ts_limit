#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:19:55 2020

@author: David
"""
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('config', help='give path to yaml-config')
args = parser.parse_args()

print(args.config)

