import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer, Flatten, BatchNormalization, LayerNormalization, Activation, Softmax, Concatenate
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D
from tensorflow.keras.regularizers import L2
from mllib.helpers import CModelConfig



# =========================================================================================================================
class CInceptionModule(layers.Layer):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_oParentModel, p_oConfig, p_nFeatures=None, p_bIsMaxPoolDownsampling=None):
        super(CInceptionModule, self).__init__()
        
        # ..................... Object Attributes ...........................
        self.Config = CModelConfig(p_oParentModel, p_oConfig)
        
        if p_nFeatures is not None:
            self.Config.Value["Convolution.Features"] = p_nFeatures
                        
        if p_bIsMaxPoolDownsampling is not None:
            self.Config.Value["IsMaxPoolDownsampling"] = p_bIsMaxPoolDownsampling
            self.Config.Value["MaxPooling.WindowSize"] = 2
            self.Config.Value["MaxPooling.Stride"] = 2
        elif "IsMaxPoolDownsampling" not in self.Config.Value:
            self.Config.Value["IsMaxPoolDownsampling"] = False
        
        # ......... Keras layers .........
        self.InputZeroPadding   = None        
        
        self.Conv1x1                     = None
        self.Conv1x1_Before3x3           = None
        self.Conv1x1_Before5x5           = None
        self.MaxPooling                  = None
        
        
        self.Conv3x3                     = None        
        self.Conv5x5                     = None
        self.Conv1x1_AfterMaxPooling     = None
        
        self.Concatenate                 = None
        
        self.Activation                  = None
        self.Normalization               = None
        self.DownsamplingMaxPooling      = None
        
        
        # ...................................................................
        
        self.Create()
    # --------------------------------------------------------------------------------------------------------
    def createWeightRegulizer(self):
        if self.Config.Value["Convolution.RegularizeL2"]:
            oWeightRegularizer = L2(self.Config.Value["Convolution.WeightDecay"])
        else:
            oWeightRegularizer = None
        return oWeightRegularizer
    # --------------------------------------------------------------------------------------------------------
    def Create(self):
        if self.Config.Value["Convolution.PaddingSize"] is not None:
            nPaddingSize = self.Config.Value["Convolution.PaddingSize"]
            if nPaddingSize > 0:
                self.InputZeroPadding   = ZeroPadding2D((nPaddingSize,nPaddingSize))
                    

                                            
        self.Conv1x1        = Conv2D(self.Config.Value["Convolution.Features"] / 2
                                        , kernel_size=1
                                        , strides=1
                                        , use_bias=True
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer()
                                        )           
        self.Conv1x1_Before3x3        = Conv2D(self.Config.Value["Convolution.Features"] / 2
                                        , kernel_size=1
                                        , strides=1
                                        , use_bias=True
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer()
                                        )   
        self.Conv1x1_Before5x5        = Conv2D(self.Config.Value["Convolution.Features"] / 2
                                        , kernel_size=1
                                        , strides=1
                                        , use_bias=True
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer()
                                        )   
        
        self.MaxPooling = MaxPooling2D(pool_size=[3,3], strides=[1, 1])
        
        
        
        
        self.Conv3x3            = Conv2D(self.Config.Value["Convolution.Features"]
                                        , kernel_size=3
                                        , strides=1
                                        , use_bias=True
                                        , padding="same"
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer()
                                        )       
        self.Conv5x5            = Conv2D(self.Config.Value["Convolution.Features"]
                                        , kernel_size=5
                                        , strides=1
                                        , use_bias=True
                                        , padding="same"
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer()
                                        ) 
        self.Conv1x1_AfterMaxPooling = Conv2D(self.Config.Value["Convolution.Features"] / 2
                                        , kernel_size=1
                                        , strides=1
                                        , use_bias=True
                                        , padding="same"
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer() 
                                        ) 
                
        self.Concatenate = Concatenate(axis=3) # Concatenation is done on the feature axis (the 4th in the rank 4 tensor)        

                                        
        self.Activation         = Activation(self.Config.Value["ActivationFunction"])
        
        if self.Config.Value["Normalization"].upper().startswith("BATCH"):
            self.Normalization      = BatchNormalization()
        elif self.Config.Value["Normalization"].upper().startswith("LAYER"):
            self.Normalization      = LayerNormalization()
        
        if "IsMaxPoolDownsampling" in self.Config.Value:
            bIsDownsampling = self.Config.Value["IsMaxPoolDownsampling"]
            if bIsDownsampling:
                nPoolSize    = self.Config.Value["MaxPooling.WindowSize"]
                nPoolStride  = self.Config.Value["MaxPooling.Stride"]
                self.DownsamplingMaxPooling  = MaxPooling2D(pool_size=[nPoolSize,nPoolSize], strides=[nPoolStride, nPoolStride])
    # --------------------------------------------------------------------------------------------------------
    def call(self, p_tInput):
        tA = p_tInput
        if self.InputZeroPadding is not None:
            tA = self.InputZeroPadding(tA)
            
            
        tA1 = self.Conv1x1(tA)
        tA2 = self.Conv1x1_Before3x3(tA)
        tA3 = self.Conv1x1_Before5x5(tA)
        tA4 = self.MaxPooling(tA)
        
        tA2 = self.Conv3x3(tA2)
        tA3 = self.Conv5x5(tA3)
        tA4 = self.Conv1x1_AfterMaxPooling(tA)
            
        tA = self.Concatenate([tA1,tA2,tA3,tA4])
        tA = self.Activation(tA)
        
        if self.Normalization is not None:
            tA = self.Normalization(tA)
            
        if self.DownsamplingMaxPooling is not None:
            tA = self.DownsamplingMaxPooling(tA)
        
        return tA 
    # --------------------------------------------------------------------------------------------------------    
    
# =========================================================================================================================












# =========================================================================================================================
class CBasicConvModule(layers.Layer):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_oParentModel, p_oConfig, p_nFeatures=None, p_bIsMaxPoolDownsampling=None):
        super(CBasicConvModule, self).__init__()
        
        # ..................... Object Attributes ...........................
        self.Config = CModelConfig(p_oParentModel, p_oConfig)
        if p_nFeatures is not None:
            self.Config.Value["Convolution.Features"] = p_nFeatures
            
        if p_bIsMaxPoolDownsampling is not None:
            self.Config.Value["IsMaxPoolDownsampling"] = p_bIsMaxPoolDownsampling
            self.Config.Value["MaxPooling.WindowSize"] = 2
            self.Config.Value["MaxPooling.Stride"] = 2
        elif "IsMaxPoolDownsampling" not in self.Config.Value:
            self.Config.Value["IsMaxPoolDownsampling"] = False
        
        
        # ......... Keras layers .........
        self.InputZeroPadding   = None        
        self.Convolution        = None
        self.Activation         = None
        self.Normalization      = None
        self.DownsamplingMaxPooling = None
        # ...................................................................
        
        self.Create()
    # --------------------------------------------------------------------------------------------------------
    def createWeightRegulizer(self):
        if self.Config.Value["Convolution.RegularizeL2"]:
            oWeightRegularizer = L2(self.Config.Value["Convolution.WeightDecay"])
        else:
            oWeightRegularizer = None
        return oWeightRegularizer        
    # --------------------------------------------------------------------------------------------------------
    def Create(self):
        
        if self.Config.Value["Convolution.PaddingSize"] is not None:
            nPaddingSize = self.Config.Value["Convolution.PaddingSize"]
            if nPaddingSize > 0:
                self.InputZeroPadding   = ZeroPadding2D((nPaddingSize,nPaddingSize))
                    
        self.Convolution        = Conv2D(self.Config.Value["Convolution.Features"]
                                        , kernel_size=self.Config.Value["Convolution.WindowSize"]
                                        , strides=self.Config.Value["Convolution.Stride"], padding="valid"
                                        , use_bias=self.Config.Value["Convolution.HasBias"]
                                        , kernel_initializer=self.Config.Value["Convolution.KernelInitializer"]
                                        , bias_initializer=self.Config.Value["Convolution.BiasInitializer"]
                                        , kernel_regularizer=self.createWeightRegulizer()
                                        )           

                                        
        self.Activation         = Activation(self.Config.Value["ActivationFunction"])
        
        if self.Config.Value["Normalization"].upper().startswith("BATCH"):
            self.Normalization      = BatchNormalization()
        elif self.Config.Value["Normalization"].upper().startswith("LAYER"):
            self.Normalization      = LayerNormalization()
        
        if "IsMaxPoolDownsampling" in self.Config.Value:
            bIsDownsampling = self.Config.Value["IsMaxPoolDownsampling"]
            if bIsDownsampling:
                nPoolSize    = self.Config.Value["MaxPooling.WindowSize"]
                nPoolStride  = self.Config.Value["MaxPooling.Stride"]
                self.DownsamplingMaxPooling  = MaxPooling2D(pool_size=[nPoolSize,nPoolSize], strides=[nPoolStride, nPoolStride])
    # --------------------------------------------------------------------------------------------------------
    def call(self, p_tInput):
        tA = p_tInput
        if self.InputZeroPadding is not None:
            tA = self.InputZeroPadding(tA)
            
        tA = self.Convolution(tA)
        
        tA = self.Activation(tA)
        
        if self.Normalization is not None:
            tA = self.Normalization(tA)
            
        if self.DownsamplingMaxPooling is not None:
            tA = self.DownsamplingMaxPooling(tA)
        
        return tA 
    # --------------------------------------------------------------------------------------------------------    
    
# =========================================================================================================================