import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mllib.utils import RandomSeed



# __________ | Settings | __________
IS_PLOTING_DATA         = False
IS_DEBUGABLE            = True
IS_RETRAINING           = True
RandomSeed(2022)

# __________ | Hyperparameters | __________
CONFIG_CNN = {
                 "ModelName": "MNIST_CNN1"
                ,"CNN.InputShape": [32,32,3]
                ,"CNN.Classes": 10
                ,"CNN.ModuleCount": 6
                ,"CNN.ConvOutputFeatures": [32,32,64,64,128,128]
                ,"CNN.ConvWindows": [ [3,2,True], [3,1,True] ,  [3,1,True], [3,2,True], [3,1,True], [3,1,True] ]
                ,"CNN.PoolWindows": [  None      , None       ,  None      , None      , [3,2]     , None      ]
                ,"CNN.HasBatchNormalization": True
                ,"Training.MaxEpoch": 24
                ,"Training.BatchSize": 128
                ,"Training.LearningRate": 0.1               
            }
                
CONFIG = CONFIG_CUSTOM_CNN
                


# __________ // Create the data objects \\ __________
import matplotlib.pyplot as plt
from datasets.cifar10.dataset import CCIFAR10DataSet



# ... // Create the data objects \\ ...
oDataset = CCIFAR10DataSet()
print("Training samples set shape:", oDataset.TSSamples.shape)
print("Validation samples set shape:", oDataset.VSSamples.shape)


if IS_PLOTING_DATA:
    for nIndex, nSample in enumerate(oDataset.TSSamples):
      nLabel = oDataset.TSLabels[nIndex]
      if nIndex == 9: 
        nImage =  nSample.astype(np.uint8) # Show the kitty
        print(nImage.shape)
        print(oDataset.ClassNames[nLabel])
        plt.imshow(nImage[4:22, 0:15, :])
        plt.show()    
    
      elif nIndex == 30: # Show the toy airplane
        nImage =  nSample.astype(np.uint8)
        print(nImage.shape)
        print(oDataset.ClassNames[nLabel])
        plt.imshow(nImage)
        plt.show()      
    
    
        plt.title("Blue")
        plt.imshow(nImage[:,:,0], cmap="Blues")
        plt.show()    
        
        plt.title("Green")
        plt.imshow(nImage[:,:,1], cmap="Greens")
        plt.show()    
    
        plt.title("Red")
        plt.imshow(nImage[:,:,2], cmap="Reds")
        plt.show()                
  
            




# -----------------------------------------------------------------------------------
def __normalizeImage(p_tImage):
    # Normalizes color component values from `uint8` to `float32`.
    return tf.cast(p_tImage, tf.float32) / 255.
# -----------------------------------------------------------------------------------
def PreprocessImageAugmentDataset(p_tImageInTS, p_tLabelInTS):
    # Normalizes color component values from `uint8` to `float32`.
    tNormalizedImage = __normalizeImage(p_tImageInTS)
    # Calls the data augmentation function that add new random samples, i.e.augments the dataset. 
    tNewRandomImage = tf.image.random_flip_left_right(tNormalizedImage)
    
    # Target class labels into one-hot encoding
    tTargetOneHot = tf.one_hot(p_tLabelInTS, CONFIG["CNN.Classes"])
    
    return tNewRandomImage, tTargetOneHot
# -----------------------------------------------------------------------------------


nBatchSize = CONFIG["Training.BatchSize"]

# Training data feed pipeline
oTSData = tf.data.Dataset.from_tensor_slices((oDataset.TSSamples, oDataset.TSLabels))
oTSData = oTSData.map(PreprocessImageAugmentDataset, num_parallel_calls=tf.data.AUTOTUNE)
oTSData = oTSData.cache()
oTSData = oTSData.shuffle(oDataset.TSSampleCount)
oTSData = oTSData.batch(nBatchSize)
oTSData = oTSData.prefetch(tf.data.AUTOTUNE)
print("Training data feed object:", oTSData)


# Validation data feed pipeline
# -----------------------------------------------------------------------------------
def PreprocessImage(p_tImageInVS, p_tLabelInVS):
    # Normalizes color component values from `uint8` to `float32`.
    tNormalizedImage = __normalizeImage(p_tImageInVS)
    # Target class labels into one-hot encoding
    tTargetOneHot = tf.one_hot(p_tLabelInVS, CONFIG["CNN.Classes"])
    
    return tNormalizedImage, tTargetOneHot
# -----------------------------------------------------------------------------------
oVSData = tf.data.Dataset.from_tensor_slices((oDataset.VSSamples, oDataset.VSLabels))
oVSData = oVSData.map(PreprocessImage, num_parallel_calls=tf.data.AUTOTUNE)
oVSData = oVSData.batch(oDataset.VSSampleCount)
print("Validation data feed object:", oVSData)





# __________ // Create the Machine Learning model and training algorithm objects \\ __________
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from models.ConvModules import CBasicConvModule, CInceptionModule
from mllib.helpers import CKerasModelStructure, CModelConfig

# =========================================================================================================================
class CMyCustomCNN(keras.Model):
    # --------------------------------------------------------------------------------------
    # Constructor
    def __init__(self, p_oConfig):
      super(CMyCustomCNN, self).__init__()
      
      # ..................... Object Attributes ...........................
      self.Config = CModelConfig(self, p_oConfig)
      
      
      self.ClassCount         = self.Config.Value["CNN.Classes"]
      self.ConvLayerFeatures  = self.Config.Value["CNN.ConvOutputFeatures"]
      self.Structure = None 
      # ......... Keras layers .........
      self.StemConv1              = None
      self.StemActivation1        = None
      self.StemBatchNorm1         = None
      
      self.StemConv2              = None
      self.StemActivation2        = None
      self.StemBatchNorm2         = None
      
      self.Module1                = None
      self.Module2                = None
      self.Module3                = None
      
      
      self.GlobalAveragePooling   = None
      self.DropOut                = None
      self.Logits                 = None
      self.SoftmaxActivation      = None
      # ...................................................................
      
      
      self.Create()
    # --------------------------------------------------------------------------------------------------------
    def createWeightRegulizer(self):
        if self.Config.Value["Training.RegularizeL2"]:
            oWeightRegularizer = regularizers.L2(self.Config.Value["Training.WeightDecay"])
        else:
            oWeightRegularizer = None
        return oWeightRegularizer          
    # --------------------------------------------------------------------------------------
    def Create(self): 
        self.StemConv1        = layers.Conv2D(self.ConvLayerFeatures[0], kernel_size=(3,3), strides=1, padding="same"
                                          , use_bias=False
                                          , kernel_initializer="glorot_uniform"
                                          , bias_initializer="zeros"
                                          , kernel_regularizer=self.createWeightRegulizer()
                                          )
        self.StemActivation1 = layers.Activation("relu")
        self.StemBatchNorm1  = layers.BatchNormalization()
        
        
        self.StemConv2        = layers.Conv2D(self.ConvLayerFeatures[1], kernel_size=(3,3), strides=2, padding="same"
                                          , use_bias=False
                                          , kernel_initializer="glorot_uniform"
                                          , bias_initializer="zeros"
                                          , kernel_regularizer=self.createWeightRegulizer()  
                                          )     
        self.StemActivation2        = layers.Activation("relu")
        self.StemBatchNorm2         = layers.BatchNormalization()
        

                
        # ..... PLACE YOUR CUSTOM ARCHITECTURE HERE .....


      
        # Using Global Average Pooling to flatten the activation tensor into an average vector
        self.GlobalAveragePooling = layers.GlobalAveragePooling2D()
        
        # Using dropout to keep 60% of the neurons randomly in each step of the training process. This mitigates overfitting. 
        self.DropOut = layers.Dropout(rate=0.4)
        
        # Output layer with class neurons that will use the SoftMax activation function    
        self.Logits = layers.Dense(self.ClassCount
                                         , use_bias=True
                                         , kernel_initializer="glorot_uniform"
                                         , bias_initializer="zeros"  
                                         , kernel_regularizer=self.createWeightRegulizer()                                   
                                   )
        self.SoftmaxActivation = layers.Softmax()           
    # --------------------------------------------------------------------------------------------------------
    def call(self, p_tInput):
        # Lazy initialization of the model structure. Will run the logic of adding keras layer to the structure just once.
        bPrint = self.Structure is None
        if bPrint:
            self.Structure = CKerasModelStructure()
            
        # ....... Stem  .......
        tA = p_tInput
        if bPrint:
            self.Structure.Add(tA)
        
        # First learnable convolutional module
        tA = self.StemConv1(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        tA = self.StemActivation1(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        tA = self.StemBatchNorm1(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        # Second learnable convolutional module
        tA = self.StemConv2(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        tA = self.StemActivation2(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        tA = self.StemBatchNorm2(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        
        
        # ....... Core  .......
        # ..... PLACE YOUR CUSTOM ARCHITECTURE HERE .....
        tA = self.Module1(tA)
        if bPrint:
            self.Structure.Add(tA)
                    
        tA = self.Module2(tA)
        if bPrint:
            self.Structure.Add(tA)        
        
        tA = self.Module3(tA)
        if bPrint:
            self.Structure.Add(tA)        
        
        # ....... Classifier  .......
        tA = self.GlobalAveragePooling(tA)
        if bPrint:
            self.Structure.Add(tA)
            
        tA = self.DropOut(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        tA = self.Logits(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        tA = self.SoftmaxActivation(tA)
        if bPrint:
            self.Structure.Add(tA)
        
        return tA
    # --------------------------------------------------------------------------------------------------------
# =========================================================================================================================



oNN = CMyCustomCNN(CONFIG)

# -----------------------------------------------------------------------------------
def LRSchedule(epoch, lr):
    nNewLR = lr
    for nIndex,oSchedule in enumerate(CONFIG["Training.LearningRateScheduling"]):
        if epoch == oSchedule[0]:
            nNewLR = oSchedule[1]
            print("Schedule #%d: Setting LR to %.5f" % (nIndex+1,nNewLR))
            break
    return nNewLR
# -----------------------------------------------------------------------------------   

nInitialLearningRate    = CONFIG["Training.LearningRate"]  
  

oCostFunction   = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
oOptimizer = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate, momentum=CONFIG["Training.Momentum"])
oCallbacks = [tf.keras.callbacks.LearningRateScheduler(LRSchedule)]



    
# __________ // Training Process \\ __________
sModelFolderName = CONFIG["ModelName"]
        
if (not os.path.isdir(sModelFolderName)) or IS_RETRAINING:
    oNN.compile(loss=oCostFunction, optimizer=oOptimizer, metrics=["accuracy"])

    oNN.predict(oVSData)
    

    oNN.Structure.Print("Model-Structure-%s.csv" % CONFIG["ModelName"])
    

    if IS_DEBUGABLE:
        oNN.run_eagerly = True
        
    oProcessLog = oNN.fit(  oTSData, batch_size=nBatchSize
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=oVSData
                            ,callbacks=oCallbacks 
                          )
    oNN.summary()          
    oNN.save(sModelFolderName)      
    
    # list all data in history
    print("Keys of Keras training process log:", oProcessLog.history.keys())
    
    
    sPrefix = sModelFolderName
            
    # summarize history for accuracy
    plt.plot(oProcessLog.history['accuracy'])
    plt.plot(oProcessLog.history['val_accuracy'])
    plt.title(sPrefix + "Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    
    sCostFunctionNameParts = oCostFunction.name.split("_")                           # [PYTHON]: Splitting string into an array of strings
    sCostFunctionNameParts = [x.capitalize() + " " for x in sCostFunctionNameParts]  # [PYTHON]: List comprehension example 
    sCostFunctionName = " ".join(sCostFunctionNameParts)                             # [PYTHON]: Joining string in a list with the space between them
    
    
    plt.plot(oProcessLog.history['loss'])
    plt.plot(oProcessLog.history['val_loss'])
    plt.title(sPrefix + sCostFunctionName + " Error")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
else:
    # The model is trained and its state is saved (all the trainable parameters are saved). We load the model to recall the samples 
    oNN = keras.models.load_model(sModelFolderName)
    oNN.summary()    

if IS_PLOTING_DATA :
    # Plot the validation set
    oPlot = CPlot("Training Set Input Features", oDataset.TSSamples[:,6:8], oDataset.TSLabels
                  ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme 
                  ,p_sXLabel="Feature 6", p_sYLabel="Feature 7" 
                  )
    oPlot.Show(p_bIsMinMaxScaled=False)
    
    
    tActivation = oNN.HiddenLayer(oDataset.TSSamples)
    nTSSamplesTransformed = tActivation.numpy()
    
    # Plot the validation set
    oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,:2], oDataset.TSLabels
                  ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme                  
                  ,p_sXLabel="Neuron 1", p_sYLabel="Neuron 2" )
    oPlot.Show(p_bIsMinMaxScaled=False)

    if nTSSamplesTransformed.shape[1] > 2:    
        oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,1:3], oDataset.TSLabels
                      ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme                      
                      ,p_sXLabel="Neuron 2", p_sYLabel="Neuron 3" )
        oPlot.Show(p_bIsMinMaxScaled=False)
        
        oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,2:4], oDataset.TSLabels
                      ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme
                      ,p_sXLabel="Neuron 3", p_sYLabel="Neuron 4" )
        oPlot.Show(p_bIsMinMaxScaled=False)

        
        
            
from mllib.evaluation import CEvaluator

      
nPredictedProbabilities = oNN.predict(oVSData)
nPredictedClassLabels  = np.argmax(nPredictedProbabilities, axis=1)


# We create an evaluator object that will produce several metrics
oEvaluator = CEvaluator(oDataset.VSLabels, nPredictedClassLabels)

oEvaluator.PrintConfusionMatrix()

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(oEvaluator.ConfusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(oEvaluator.ConfusionMatrix.shape[0]):
    for j in range(oEvaluator.ConfusionMatrix.shape[1]):
        ax.text(x=j, y=i,s=oEvaluator.ConfusionMatrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('Actual Label', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



print("Per Class Recall (Accuracy)  :", oEvaluator.Recall)
print("Per Class Precision          :", oEvaluator.Precision)
print("Average Accuracy: %.4f" % oEvaluator.AverageRecall)
print("Average F1 Score: %.4f" % oEvaluator.AverageF1Score)
      




# Plot the error after the training is complete
#oTrainingError = np.asarray(oMeanError, dtype=np.float64)
#plt.plot(oMeanError)
#plt.show()
