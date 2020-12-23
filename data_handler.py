import pandas as pd
import numpy as np
import os

from util import Singleton
from util import sigmoid

class DataHandler(metaclass=Singleton):
  def __init__(self, filepaths, results_path, csv_sep):
    self.filepaths = filepaths
    self.results_path = results_path
    self.csv_sep = csv_sep
  
  def save_sentiment_to_files(self, scores_of_files):
    if not os.path.exists(self.results_path):
      os.mkdir(self.results_path)
    
    for filepath, df, sentiment_scores in zip(self.filepaths, self.dfs, scores_of_files):
      for row, sentiment_score in enumerate(sentiment_scores):
        df.loc[row, 'sentiment'] = sentiment_score

        # Zero values indicate non valid sentiment
        if (sentiment_score != 0):
          sentiment_score += self.__calculate_karma(df.loc[row, 'score'], 0.01)

      df.to_csv(f'./results/{os.path.basename(filepath)}_out.csv', sep=self.csv_sep)

  def get_text_from_files(self):
    """returns all the comments/text included in the csv files
        
      Parameters: None
        
      Return:
      output: list
          2D list containing the comments/text of the each file respectably
    """

    self.dfs = []
    for filepath in self.filepaths:
      self.dfs.append(self.__open_file(filepath))

    files_text = []
    for df in self.dfs:
      files_text.append(df['body'])
  
    return files_text
  
  # Rate changes how much karma affect the total sentiment
  # greater rate means greater sentiment
  def __calculate_karma(self, score, rate):
    ## Do some clever stuff
    return sigmoid(score * rate)

  
  def __open_file(self, filepath):
    return pd.read_csv(filepath, sep=self.csv_sep)
