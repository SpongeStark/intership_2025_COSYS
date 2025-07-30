"""
File: tools.py
Author: YANG Kai
Date: 2025-03-28
Description: All the tool functions
"""

def weighted_mean_for_last_layer(values, weight_for_last_layer):
    n = len(values)
    w = (1 - weight_for_last_layer) / (n-1)
    weights = [w]*(n-1)+[weight_for_last_layer]
    return sum([values[i]*weights[i] for i in range(n)])


if __name__=="__main__":
    print(weighted_mean_for_last_layer([1,1,2], 0.5))