

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class data_split_train_test:
  
  def __init__(self, df):
    self.df = df
    
  def split_data_train_test_random(self, train_proportion, test_proportion, random_state=42):
    df_train, df_test = train_test_split(self.df, train_size=train_proportion, test_size=test_proportion, random_state=42)
    return df_train, df_test
  
#   def split_data_train_test_user_stratified