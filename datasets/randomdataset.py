import numpy as np                      # use the package (a.k.a. namespace) with the alias "np"
from sklearn import datasets            # import as single object/subpackage from the package
from sklearn.model_selection import train_test_split    # import a standalone procedure function from the pacckage
from mllib.utils import RandomSeed                      # custom python package, file inside the package.
 

# ====================================================================================================
class CRandomDataset(object):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nSampleCount=200, p_nClustersPerClass=1, p_nClassSeperability=1.0, p_nRandomSeed=2022):
    # ................................................................
    # // Fields \\
    self.RandomSeed = p_nRandomSeed
    self.Samples   = None
    self.Labels    = None
    self.SampleCount = p_nSampleCount

    self.TSSamples = None
    self.TSLabels  = None
    self.TSSampleCount = 0

    self.VSSamples = None
    self.VSLabels  = None
    self.VSSampleCount = 0
    # ................................................................

    RandomSeed(self.RandomSeed )
    self.Samples, self.Labels = datasets.make_classification(
        n_features=2,
        n_classes=2,
        n_samples=self.SampleCount,
        n_redundant=0,
        n_clusters_per_class=p_nClustersPerClass,
        class_sep=p_nClassSeperability
        
    )
  # --------------------------------------------------------------------------------------
  # Method 
  def DebugPrint(self):
    print("Shape of sample matrix", self.Samples.shape)
    print('.'*80)

    print("Datatype of sample matrix before convertion: %s" % str(self.Samples.dtype))
    # Convert the data to 32bit floating point numbers (default for faster computations)
    self.Samples = np.asarray(self.Samples, dtype=np.float32)
    print("Datatype of sample matrix after convertion: %s" % str(self.Samples.dtype))
    print('.'*80)

    # Classification into 2 classes == Binary classification
    print("Class labels")
    print(self.Labels)
    print('.'*80)
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
# ====================================================================================================      