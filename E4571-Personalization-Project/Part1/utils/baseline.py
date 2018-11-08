
import pandas as pd
import numpy as np


class baseline:
  
  def __init__(self, df, user_field, item_field, rec_field):
    self.df = df
    self.user_field = user_field
    self.item_field = item_field
    self.rec_field = rec_field
    self.create_overall_average()
    self.create_user_average()
    self.create_item_average()
    self.create_item_adj_index()
    
    
  def create_overall_average(self):
    self.complete_avg = self.df[self.rec_field].mean()
  
  def create_user_average(self):
    self.user_avg = self.df.groupby(self.user_field)[self.rec_field].mean()
    
  def create_item_average(self):
    self.item_avg = self.df.groupby(self.item_field)[self.rec_field].mean()
    
  def create_item_adj_index(self):
    self.item_avg_mean = self.item_avg.mean()
    self.item_avg_index = self.item_avg/self.item_avg_mean
    
  def baseline_average(self, user, item):
    return self.complete_avg
  
# Account for the fact that the user might not be part of the training dataset whatsoever
  def baseline_user_average(self, user, item):
    return self.user_avg[user]
    
  def baseline_item_average(self, user, item):
    return self.item_avg[item]
  
  def baseline_user_item_adjusted(self, user, item):
    return self.user_avg[user] * self.item_avg_index[item]