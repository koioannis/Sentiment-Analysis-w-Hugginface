import pandas as pd
import numpy as np

from util import Singleton

class DataHandler(metaclass=Singleton):
  def __init__(self, filepaths):
    self.filepaths = filepaths
  
  def __open_file(self, filepath):
   return pd.read_csv(filepath, sep=";")
  
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
      files_text.append(df["body"])

    return files_text

        