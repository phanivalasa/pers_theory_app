
import pandas as pd
import numpy as np

def mean_squared_error(actual_arr, pred_arr):
    return np.average(np.power(actual_arr-pred_arr,2))

def root_mean_squared_error(actual_arr, pred_arr):
    return np.sqrt(np.average(np.power(actual_arr-pred_arr,2)))
  
def mean_absolute_error(actual_arr, pred_arr):
    return np.average(np.abs(actual_arr-pred_arr))