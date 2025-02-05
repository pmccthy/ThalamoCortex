"""
Utility functions.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""


from itertools import product

def make_grid(param_dict):  
    param_names = param_dict.keys()
    combinations = product(*param_dict.values()) # creates list of all possible combinations of items in input list
    ds=[dict(zip(param_names, param_val)) for param_val in combinations] # convert to list of dicts
    return ds