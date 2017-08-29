'''
###########################################
#                                         # 
# This class provides foolproof pca calls.#
#                                         #
###########################################
Usage:
   Given a numpy array or pure number Dataframe X.
pca = upca()
pca.fit(X)
xp = pca.transform(X)
'''
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn import preprocessing
import glob
import os
import pickle
import sys

def to_array(X):
    '''
    This function will try to convert X to a numpy array
    and check whether it's valid (containing only numbers).
    '''
    try:
        X = np.array(X)
        assert(np.isfinite(X).all()), \
              "Input with %s type is not valid numpy array of finite numbers." % type(X)
        return X
    except AttributeError:
        print "Invalid input that can't be converted to numpy array."
        sys.exit(1)
    
def check_standardized(X):
    '''
    Check whether X is standardized
    '''
    std = X.std(axis=0)
    mean = X.mean(axis=0)
    ones = np.ones(len(std))
    zeros = np.zeros(len(mean))
    return ( np.isclose(std, ones, atol=0.1).all() and np.isclose(mean, zeros, atol=.1).all() )
        
class upca(object):
  def __init__(self):
    self.model = None

  def fit(self, X, STANDARDIZE=True, n=10):
      if not isinstance(X, np.ndarray):
          X =  to_array(X)
      assert(X.ndim == 2), "Input array must have two dimensions."
      if not check_standardized(X):
          if STANDARDIZE:
              X = preprocessing.scale(X)
              print "Standardize input data for fit."
          else:
              print "WARNING: data is not standardized and you switch off STANDARDIZE option.",
              print "Make sure this is what you intended."
      self.model = PCA(n_components=n)
      self.model.fit(X)
        
            
  def load(self, modelfile):
      with open(modelfile, 'rb') as pklfile:
          self.model = pickle.load(pklfile)
      assert( isinstance(self.model,sklearn.decomposition.pca.PCA) )
      print "Successfully loaded pca model from %s." % modelfile
      
  def save(self, modelfile):
      with open(modelfile, 'wb') as pklfile:
          pickle.dump(self.model, pklfile)
      print "Save model to %s" % modelfile
      
  def transform(self, X, STANDARDIZE=True):
      if not isinstance(X, np.ndarray):
          X =  to_array(X)
      assert(X.ndim == 2), "Input array must have two dimensions."
      if not check_standardized(X):
          if STANDARDIZE:
              X = preprocessing.scale(X)
              print "Standardize input data for transform"
      if not self.model:
          print "Load or fit a model before performing trsnaformation."
      else:
          assert(X.shape[1] > self.model.n_components),\
              "Input data must have a dimension larger than model components %d."\
              % self.model.n_components
          xp = self.model.transform(X)
          return xp
      
if __name__ =="__main__":
    pca  = upca()
    pca.load('pca1.pkl')
    a = np.arange(10000)
    a = a.reshape(100,100)
    a = preprocessing.scale(a)
    pca.fit(a)
    xp = pca.transform(a)
