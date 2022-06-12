# ......................................................................................
# MIT License

# Copyright (c) 2021 Pantelis I. Kaplanoglou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ......................................................................................

import pickle
import numpy as np
import sys
import os
import scipy.stats as stats
from mllib.filestore import CFileStore
from mllib.data import CCustomDataSet
#from rx.statistics import CHistogramOfClasses
from datasets.cifar10.downloader import CDataSetDownloaderCIFAR10


# =========================================================================================================================
class CCIFAR10DataSet(CCustomDataSet):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_bIsVerbose=False):
    super(CCIFAR10DataSet, self).__init__()
    # ................................................................
    # // Fields \\
    self.IsVerbose = p_bIsVerbose

    self.DataSetFolder = os.path.join("MLData", "cifar10")

    self.ClassCount         = 10
    self.ClassNames         = [  "airplane", "automobile", "bird", "cat","deer"
                               , "dog", "frog", "horse", "ship", "truck"
                               ]
    self.FeatureCount       = 32*32*3
    self.ImageShape         = [32, 32, 3]


    self.BatchesFile                = os.path.join(self.DataSetFolder, 'batches.meta')
    self.TrainingShardFileTemplate  = os.path.join(self.DataSetFolder, 'data_batch_%d')
    self.TestFileName               = os.path.join(self.DataSetFolder, 'test_batch')

    
    self.FileStore = CFileStore(self.DataSetFolder)
    # ................................................................

    # Lazy dataset initialization. Try to load the data and if not already cached to local filestore, generate the samples now and cache them.
    self.TSSamples = self.FileStore.Deserialize("CIFAR10-TSSamples.pkl")
    self.TSLabels  = self.FileStore.Deserialize("CIFAR10-TSLabels.pkl")

    self.VSSamples = self.FileStore.Deserialize("CIFAR10-VSSamples.pkl")
    self.VSLabels  = self.FileStore.Deserialize("CIFAR10-VSLabels.pkl")

    if self.TSSamples is None:
      self.CreateDatasetCache()

      self.FileStore.Serialize("CIFAR10-TSSamples.pkl", self.TSSamples)
      self.FileStore.Serialize("CIFAR10-TSLabels.pkl", self.TSLabels)
      
      self.FileStore.Serialize("CIFAR10-VSSamples.pkl", self.VSSamples)
      self.FileStore.Serialize("CIFAR10-VSLabels.pkl", self.VSLabels)
    else:
      self.TSSampleCount    = self.TSSamples.shape[0]
      self.VSSampleCount    = self.VSSamples.shape[0]
      self.SampleCount      = self.TSSampleCount + self.VSSampleCount
      self.FeatureCount     = np.prod(self.TSSamples.shape[1:])
      self.ClassCount       = len(np.unique(self.TSLabels))



  # --------------------------------------------------------------------------------------
  def CreateDatasetCache(self):
    oDownloader = CDataSetDownloaderCIFAR10()
    oDownloader.Download()

    self.LoadSubset(True)
    self.LoadSubset(False)
    
    self.SampleCount      = self.TSSampleCount + self.VSSampleCount
    self.FeatureCount     = np.prod(self.TSSamples.shape[1:])
    self.ClassCount       = len(np.unique(self.TSLabels))

    print("Classes:", self.ClassCount)
  # --------------------------------------------------------------------------------------------------------
  def AppendTrainingShard(self, p_nSamples, p_nLabels):
    # First shard initializes training set, next shards are appended
    if self.TSSamples is None:
      self.TSSamples = p_nSamples
      self.TSSampleCount = 0
    else:
      self.TSSamples = np.concatenate((self.TSSamples, p_nSamples), axis=0)

    if self.TSLabels is None:
      self.TSLabels = p_nLabels
    else:
      self.TSLabels = np.concatenate((self.TSLabels, p_nLabels), axis=0)
      
    self.TSSampleCount += p_nSamples.shape[0]
  # --------------------------------------------------------------------------------------------------------
  def AppendValidationShard(self, p_nSamples, p_nLabels):
    # First shard initializes test (validation) set, next shards are appended
    if self.VSSamples is None:
      self.VSSamples = p_nSamples
      self.VSSampleCount = 0
    else:
      self.VSSamples = np.concatenate((self.VSSamples, p_nSamples), axis=0)

    if self.VSLabels is None:
      self.VSLabels = p_nLabels
    else:
      self.VSLabels = np.concatenate((self.VSLabels, p_nLabels), axis=0)        

    self.VSSampleCount += p_nSamples.shape[0]
  # --------------------------------------------------------------------------------------------------------
  def _transposeImageChannels(self, p_nX, p_nShape=(32, 32, 3), p_bIsFlattening=False):
    """
    This method create image tensors (Spatial_dim, Spatial_dim, Channels) from image vectors of 32x32x3 features
    """
    nResult = np.asarray(p_nX, dtype=np.float32)
    nResult = nResult.reshape([-1, p_nShape[2], p_nShape[0], p_nShape[1]])
    nResult = nResult.transpose([0, 2, 3, 1])
        
    if p_bIsFlattening:
      nResult = nResult.reshape(-1, np.prod(np.asarray(p_nShape)))
        
    return nResult 
  # --------------------------------------------------------------------------------------------------------
  def LoadSubset(self, p_bIsTrainingSet=True):
    if p_bIsTrainingSet:
      for i in range(5):
        with open(self.TrainingShardFileTemplate % (i+1), 'rb') as oFile:
          oDict = pickle.load(oFile, encoding='latin1')
          oFile.close()
        self.AppendTrainingShard(self._transposeImageChannels(oDict["data"], (32,32,3)), np.array(oDict['labels'], np.uint8))
    else:
      with open(self.TestFileName, 'rb') as oFile:
        oDict = pickle.load(oFile, encoding='latin1')
        oFile.close()
      self.AppendValidationShard(self._transposeImageChannels(oDict["data"], (32,32,3)), np.array(oDict['labels'], np.uint8))
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


if __name__ == "__main__":
  oDataSet = CCIFAR10DataSet()
  print(oDataSet.TSSamples.shape)
  print(oDataSet.VSSamples.shape)
