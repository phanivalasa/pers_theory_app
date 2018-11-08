
import pandas as pd
import numpy as np
import scipy
from utils.baseline import baseline

# Right now, the similarity matrix gets precomputed
# In the future, maybe have the ability to not precompute the similarity matrix but let it be more of a lazy evaluation
class get_neighbors:
  
  def __init__(self, df, user_field, item_field, rec_field, k):
    self.df = df
    self.user_field = user_field
    self.item_field = item_field
    self.rec_field = rec_field
    self.k = k
    self.similarity_function_lookup = {'euclidian': self._euclidian_dist,
                                       'pearson': self._pearson_corr,
                                       'cosine': self._cosine_dist,
                                       'cosine_adjusted': self._cosine_dist,
                                      }
    
    self.compute_matrix()

  def compute_matrix(self):
    self.df_matrix = self.df.pivot(index=self.user_field, columns=self.item_field, values=self.rec_field)

  def compute_user_adjusted_matrix(self):
    df_copy = self.df.copy()
    df_copy[self.rec_field+'_user_mean'] = df_copy.groupby(self.user_field)[self.rec_field].transform(lambda x:np.mean(x))
    df_copy[self.rec_field+'_user_adj'] = df_copy[self.rec_field] - df_copy[self.rec_field+'_user_mean']
    self.df_matrix_adjusted = df_copy.pivot(index=self.user_field, columns=self.item_field, values=self.rec_field+'_user_adj').reset_index()

    
  @staticmethod  
  def _compute_intersection(arr1, arr2):
    arr1_nonzero = arr1.fillna(0).nonzero()[0]
    arr2_nonzero = arr2.fillna(0).nonzero()[0]
    intersection = np.intersect1d(arr1_nonzero, arr2_nonzero)
    return arr1.take(intersection), arr2.take(intersection)
  
  @classmethod  
  def _euclidian_dist(cls, arr1, arr2):
    arr1, arr2 = cls._compute_intersection(arr1, arr2)
    dist = np.sqrt(np.sum((arr1 - arr2)**2))
    return 1/(1+dist)
  
  @classmethod  
  def _pearson_corr(cls, arr1, arr2):
    arr1, arr2 = cls._compute_intersection(arr1, arr2)
    return scipy.stats.pearsonr(arr1, arr2)[0]
  
  @classmethod  
  def _cosine_dist(cls, arr1, arr2):
    arr1, arr2 = cls._compute_intersection(arr1, arr2)
    return 1 - scipy.spatial.distance.cosine(arr1, arr2)


  def compute_similarity(self, similarity_measure):
    similarity_function = self.similarity_function_lookup[similarity_measure]
    self.itemset = self.df[self.item_field].unique()    
    sim_matrix = np.zeros([len(self.itemset), len(self.itemset)])
    
    if similarity_measure =='cosine_adjusted':
      self.compute_user_adjusted_matrix()
      df_matrix_selected = self.df_matrix_adjusted
    else: df_matrix_selected = self.df_matrix
      
    for ctr1, val1 in enumerate(self.itemset):
      for ctr2, val2 in enumerate(self.itemset):
        if val1 != val2:
          sim_matrix[ctr1, ctr2] = similarity_function(df_matrix_selected[val1], df_matrix_selected[val2])
    return sim_matrix
  
  def get_neighbors(self, similarity_measure):
    sim_matrix = self.compute_similarity(similarity_measure)
    neighbors = {}
    for item in self.itemset:
      all_neighbors_scores = sim_matrix[int(np.where(self.itemset==item)[0])] # This line is only required to make this reusable for individual neighbors lookup

      all_neighbors_scores_sorted = np.sort(all_neighbors_scores)[::-1]
      all_neighbors_index_sorted = np.argsort(all_neighbors_scores)[::-1]

      neighbors_index = all_neighbors_index_sorted[:self.k]
      neighbor_items = np.take(self.itemset, neighbors_index)
      neighbor_scores = all_neighbors_scores_sorted[:self.k]

      neighbors[item] = list(zip(neighbor_items, neighbor_scores))

    return neighbors



class ItembasedCF:
  
  def __init__(self, k, similarity_measure, baseline='average'):
    self.k = k
    self.similarity_measure = similarity_measure
    self.baseline = baseline
    
    
  # Pass only the training set to this function
  def fit(self, df, user_field, item_field, rec_field):
    self.user_field = user_field
    self.item_field = item_field
    self.rec_field = rec_field
    self.userset = df[user_field].unique()
    sim = get_neighbors(df, user_field=user_field, item_field=item_field, rec_field=rec_field, k=self.k)
    self.neighbors = sim.get_neighbors(self.similarity_measure)
    self.df_matrix = sim.df_matrix
    
    self.baseline_pred = baseline(df, user_field=user_field, item_field=item_field, rec_field=rec_field)
    
    
    
  def predict_individual(self, user, item):
    
    # Looking up the baseline function before moving onto the prediction
    
    self.baseline_function_lookup = {'average': self.baseline_pred.baseline_average,
                                       'user_average': self.baseline_pred.baseline_user_average,
                                       'item_average': self.baseline_pred.baseline_item_average,
                                       'user_item_adjusted': self.baseline_pred.baseline_user_item_adjusted
                                      }

    baseline_func = self.baseline_function_lookup[self.baseline]
    
    if user not in self.userset:
      pred = baseline_func(user, item)
      return (pred,1)

    item_neighbors = self.neighbors[item]
      
    item_neighbor_ratings = [(item, score, self.df_matrix.loc[user][item]) for (item, score) in item_neighbors]
    weighted_sum = sum([score*rec_field for (item, score, rec_field) in item_neighbor_ratings if not np.isnan(rec_field)])
    sum_of_weights = sum([score for (item, score, rec_field) in item_neighbor_ratings if not np.isnan(rec_field)])
    
    
    
    
    if weighted_sum==0:
      pred = baseline_func(user, item)
      return (pred,1)
    
    else: pred = weighted_sum/sum_of_weights
      
    return (pred,0)
  
  
  def predict(self, df, baseline_col=False):
    if baseline_col:
      pred_series = df.apply(lambda x:self.predict_individual(x[self.user_field], x[self.item_field]), axis=1).apply(pd.Series)
      df['pred'] = pred_series[0]
      df['baseline_col'] = pred_series[1]
      return df
    else:
      df['pred'] = df.apply(lambda x:self.predict_individual(x[self.user_field], x[self.item_field])[0], axis=1)
    return df
    
    
#   def evaluate(self, df):
#     We also want to understand what percentage of the predictions fell into the baseline
    
#   def recommend_similar_items():
    