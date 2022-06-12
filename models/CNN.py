import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Flatten, Dense, BatchNormalization, Activation, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D  
from tensorflow.keras.regularizers import L2
from mllib.helpers import CKerasModelStructure

# =========================================================================================================================
class CCNNBasic(keras.Model):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_oConfig):
    super(CCNNBasic, self).__init__()
    
    # ..................... Object Attributes ...........................
    self.Config = p_oConfig
    
    self.InputShape         = self.Config["CNN.InputShape"]
    self.ClassCount         = self.Config["CNN.Classes"]
    self.ModuleCount        = self.Config["CNN.ModuleCount"]
    
    self.ConvLayerFeatures  = self.Config["CNN.ConvOutputFeatures"]
    self.ConvWindows        = self.Config["CNN.ConvWindows"]
    self.PoolWindows        = self.Config["CNN.PoolWindows"]
    
    if "CNN.HasBatchNormalization" not in self.Config:
        self.Config["CNN.HasBatchNormalization"] = False
    
    self.KerasLayers        = []

    self.OutputLayer        = None
    self.SoftmaxActivation  = None
    self.Input              = None
    self.Structure          = None
    # ...................................................................
    
    # Default values for extra customization
    
    if "CNN.ActivationFunction" not in self.Config:
        self.Config["CNN.ActivationFunction"] = "relu"
                
    if "CNN.ConvHasBias" not in self.Config:
        self.Config["CNN.ConvHasBias"] = False

    if "CNN.KernelInitializer" not in self.Config:
        self.Config["CNN.KernelInitializer"] = "glorot_uniform"

    if "CNN.BiasInitializer" not in self.Config:
        self.Config["CNN.BiasInitializer"] = "zeros"

    if "Training.RegularizeL2" not in self.Config:
        self.Config["Training.RegularizeL2"] = False
                 
    if "Training.WeightDecay" not in self.Config:
        self.Config["Training.WeightDecay"] =  1e-5
        
    if self.Config["Training.RegularizeL2"]:
        print("Using L2 regularization of weights with weight decay %.6f" % self.Config["Training.WeightDecay"])

                                    
    self.Create()
  # --------------------------------------------------------------------------------------
  def Create(self):                # override a virtual in our base class
    # This loop creates stacked convolutional modules of the form   CONVOLUTION - ACTIVATION - NORMALIZATION - MAX POOLING
    for nModuleIndex in range(0, self.ModuleCount):
      nFeatures     = self.ConvLayerFeatures[nModuleIndex]
      oConvWindowSetup = self.ConvWindows[nModuleIndex]
      nWindowSize   = oConvWindowSetup[0]
      nStride       = oConvWindowSetup[1]
      
      sPaddingType      = "valid"
      if len(oConvWindowSetup) == 3:
          bIsPadding    = oConvWindowSetup[2]
          if bIsPadding:
              sPaddingType = "same"
      
      if self.Config["Training.RegularizeL2"]:
          oWeightRegularizer = L2(self.Config["Training.WeightDecay"])
      else:
          oWeightRegularizer = None
                        
      oConvolution = Conv2D(nFeatures, kernel_size=nWindowSize, strides=nStride, padding=sPaddingType
                            , use_bias=self.Config["CNN.ConvHasBias"]
                            , kernel_initializer=self.Config["CNN.KernelInitializer"]
                            , bias_initializer=self.Config["CNN.BiasInitializer"]
                            , kernel_regularizer=oWeightRegularizer                            
                            )
      self.KerasLayers.append(oConvolution)
      
      oActivation  = Activation(self.Config["CNN.ActivationFunction"])
      self.KerasLayers.append(oActivation)
      
      if self.Config["CNN.HasBatchNormalization"]:
          oNormalization = BatchNormalization()
          self.KerasLayers.append(oNormalization)
      
      oPoolWindow   = self.PoolWindows[nModuleIndex]
      # Set the pool size to None for a module that does not do Max Pooling.
      if oPoolWindow is not None:
          nPoolSize   = oPoolWindow[0]
          nPoolStride = oPoolWindow[1]
          oMaxPooling = MaxPooling2D(pool_size=[nPoolSize, nPoolSize], strides=[nPoolStride, nPoolStride])
          self.KerasLayers.append(oMaxPooling)
          
    
    # After the stack of convolutional modules, the activation cube will be flattened to a vector using a Flatten keras layer
    self.FlatteningLayer = Flatten()
    
    
    # The output layer for the classifier is a fully connected (dense) that has one neuron for each class.
    # You might consider the stack of convolutional modules functioning as the "hidden" layer in the 2-layer NN architecture.
    if self.Config["Training.RegularizeL2"]:
        oWeightRegularizer = L2(self.Config["Training.WeightDecay"])
    else:
        oWeightRegularizer = None          
    self.OutputLayer = Dense(self.ClassCount, use_bias=True
                             ,kernel_initializer=self.Config["CNN.KernelInitializer"]
                             ,bias_initializer=self.Config["CNN.BiasInitializer"]
                             ,kernel_regularizer=oWeightRegularizer                                 
                             )
    
    # Instead of using sigmoid for each neuron, we use the softmax activation function so that neuron "fire" together. 
    self.SoftmaxActivation = Softmax()           
  # --------------------------------------------------------------------------------------------------------
  def call(self, p_tInput):        # overrides a virtual in keras.Model class
    bPrint = self.Structure is None
    if bPrint:
        self.Structure = CKerasModelStructure()
      
    self.Input = p_tInput
    
    # ....... Convolutional Feature Extraction  .......
    # Feed forward to the next layer
    tA = p_tInput
    for nIndex,oKerasLayer in enumerate(self.KerasLayers):
        if bPrint:
            self.Structure.Add(tA)         
        tA = oKerasLayer(tA)

    # Flattens the activation cube to a vector
    tA = self.FlatteningLayer(tA)
    if bPrint:
        self.Structure.Add(tA)         
    
    # ....... Classifier  .......
    # Fully connected (dense) layer that has a count of neurons equal to the classes, with softmax activation function
    tA = self.OutputLayer(tA)
    if bPrint:
        self.Structure.Add(tA)         
    
    tA = self.SoftmaxActivation(tA)
    if bPrint:
        self.Structure.Add(tA)         
        
    
    return tA
  # --------------------------------------------------------------------------------------
# =========================================================================================================================