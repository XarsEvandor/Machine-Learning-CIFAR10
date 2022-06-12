# ......................................................................................
# MIT License

# Copyright (c) 2020-2022 Pantelis I. Kaplanoglou

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

from sklearn.model_selection import train_test_split    # import a standalone procedure function from the pacakge
from mllib.utils import RandomSeed

# =========================================================================================================================
class CCustomDataSet(object):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nSampleCount=None, p_nFeatureCount=None, p_nClassCount=None, p_nRandomSeed=None):
    # ................................................................
    # // Fields \\
    self.FeatureCount       = p_nFeatureCount
    self.ClassCount         = p_nClassCount
    self.SampleCount        = p_nSampleCount
    self.Samples            = None
    self.Labels             = None
    

    self.TSSamples          = None
    self.TSLabels           = None
    self.TSSampleCount      = 0

    self.VSSamples          = None
    self.VSLabels           = None
    self.VSSampleCount      = 0
    
    self.RandomSeed = p_nRandomSeed
    # ................................................................
    if self.RandomSeed is not None:
        RandomSeed(self.RandomSeed)
  # --------------------------------------------------------------------------------------
  def Split(self, p_nValidationSetPercentage):
    self.TSSamples, self.VSSamples, self.TSLabels, self.VSLabels = train_test_split(
                                                              self.Samples, self.Labels
                                                            , test_size=p_nValidationSetPercentage
                                                            , random_state=self.RandomSeed)
        
    
    self.TSSampleCount = self.TSSamples.shape[0]
    self.VSSampleCount = self.VSSamples.shape[0]
    
    
    print("%d ssamples in the Training Set" % self.TSSampleCount)
    print("%d ssamples in the Validation Set"%  self.VSSampleCount)
    print('.'*80)
  # --------------------------------------------------------------------------------------
# =========================================================================================================================
