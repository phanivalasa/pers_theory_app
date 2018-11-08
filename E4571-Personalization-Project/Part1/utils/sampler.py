# Currently takes the top n users, then takes the top m items among the n users and then gets the intersection set
# Currently gets the top n users by item counts (not listening counts) - just how many different songs they listen to
# Need to pass in the pandas dataframe that contains the user, item, and counts (or ratings) and the corresponding column names
# Also need to pass in the number of users and items for the sampling

import pandas as pd
import numpy as np


class rec_sample:
  
  def __init__(self, df, user_field, item_field):
    self.user_field = user_field
    self.item_field = item_field
    self.df = df
    
  def generate_sample(self, users_n, items_m):
    user_item_count = self.df.groupby(self.user_field)[self.item_field].count().reset_index()
    user_item_count.sort_values(self.item_field, inplace=True, ascending=False)
    top_users = user_item_count[:users_n][[self.user_field]]

    top_users_df = pd.merge(self.df, top_users, on=self.user_field, how='inner')
    
    item_user_count = top_users_df.groupby(self.item_field)[self.user_field].count().reset_index()
    item_user_count.sort_values(self.user_field, ascending=False, inplace=True)
    top_items = item_user_count[:items_m][[self.item_field]]
    
    top_users_items_df = pd.merge(top_users_df, top_items, on=self.item_field, how='inner')
    
#     self.df_sampled_shape = top_users_items_df.shape
        
    return top_users_items_df    