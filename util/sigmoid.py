import numpy as np

def sigmoid(x):
  """ applies sigmoid to an input x"""
  return 1/(1 + np.exp(-x)) 