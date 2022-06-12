## Changed the number of epochs to 500. 
* <details><summary>Modules</summary>
    <p>

        ```python

        self.Module1 = CInceptionModule(self, oCommonModuleConfig, 64,p_bIsMaxPoolDownsampling=True)
            # self.DropoutConv = layers.Dropout(rate = 0.4)                                               #Make rate a self.Config.Value["CNN.DropoutProbability"]
            self.Module2 = CInceptionModule(self, oCommonModuleConfig, 64,p_bIsMaxPoolDownsampling=True)
            self.Module3 = CInceptionModule(self, oCommonModuleConfig, 64)
        
        ```

    </p>
    </details>

* ### Too many epochs, the model overfits early.

## Same number of epochs, removed the custom modules.
* ### The model plateaus at 60% validation accuracy


## Reduced epochs to 100 and increased stride to 2.
* ### 60% accuracy. A lot of oscilation.

## 100 Epochs, 2 Stride, Custom modules.
* <details><summary>Modules</summary>
    <p>

        ```python

        self.Module1 = CInceptionModule(self, oCommonModuleConfig, 64,p_bIsMaxPoolDownsampling=True)
            # self.DropoutConv = layers.Dropout(rate = 0.4)                                               #Make rate a self.Config.Value["CNN.DropoutProbability"]
            self.Module2 = CInceptionModule(self, oCommonModuleConfig, 64,p_bIsMaxPoolDownsampling=True)
            self.Module3 = CInceptionModule(self, oCommonModuleConfig, 64)
        
        ```

    </p>
    </details>
* ### 76% accuracy. A lot of oscilation. HUGE overfitting.


## Changed Stem architecture.
* <details><summary>Architecture</summary>
    <p>

        ```python
        "CNN.ConvOutputFeatures": [32,32,64,128,256]

        self.Module1 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[0],p_bIsMaxPoolDownsampling=True)
        self.DropoutConv = layers.Dropout(rate = 0.4)                         #Make rate a self.Config.Value["CNN.DropoutProbability"]
        self.Module2 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[1])
        self.Module3 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[2],p_bIsMaxPoolDownsampling=True)
        self.Module4 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[3])
        self.Module5 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[4])
        ```

    </p>
    </details>

* ### reached 80% accuracy before i killed it for being too slow. some oscilation and overfitting. Plateaued.

## Removed the dropout layer.
* ### Worse overall performance

## Added the dropout layer again but changed the downsampling location to module 4
* <details><summary>Architecture</summary>
    <p>

        ```python
        "CNN.ConvOutputFeatures": [32,32,64,128,256]

        self.Module1 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[0],p_bIsMaxPoolDownsampling=True)
        self.DropoutConv = layers.Dropout(rate = 0.4)                         #Make rate a self.Config.Value["CNN.DropoutProbability"]
        self.Module2 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[1])
        self.Module3 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[2],p_bIsMaxPoolDownsampling=True)
        self.Module4 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[3])
        self.Module5 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[4])
        ```

    </p>
    </details>

* ### Plateaus at about 80-82% validation. Overfitting.


## Added more dropout layers. Reduced epochs to 30 for easier testing.
* <details><summary>Architecture</summary>
    <p>

        ```python
        tA = self.Module1(tA)
        if bPrint:
            self.Structure.Add(tA)

        tA = self.Module2(tA)
        if bPrint:
            self.Structure.Add(tA)
            
        tA = self.DropoutConv(tA)
        if bPrint:
            self.Structure.Add(tA)
            
        tA = self.Module3(tA)
        if bPrint:
            self.Structure.Add(tA)

        tA = self.DropoutConv(tA)
        if bPrint:
            self.Structure.Add(tA)

        tA = self.Module4(tA)
        if bPrint:
            self.Structure.Add(tA)

        tA = self.Module5(tA)
        if bPrint:
            self.Structure.Add(tA)
        ```

    </p>
    </details>

* ### 80% Accuracy, large overfitting.


## Changed the learning rate scheduler and the downsampling rate. Changed epochs to 50.
* <details><summary>Architecture</summary>
    <p>

        ```python
        oCommonModuleConfig={  "Convolution.Features"           : None 
                              ,"Convolution.PaddingSize"        : 1
                              ,"Convolution.WindowSize"         : 3
                              ,"Convolution.Stride"             : 1
                              ,"Convolution.KernelInitializer"  : "glorot_uniform"
                              ,"Convolution.HasBias"            : False
                              ,"Convolution.BiasInitializer"    : None
                              ,"Convolution.RegularizeL2"       : self.Config.Value["Training.RegularizeL2"]
                              ,"Convolution.WeightDecay"        : self.Config.Value["Training.WeightDecay"]
                              ,"ActivationFunction"             : "relu"
                              ,"Normalization"                  : "BatchNormalization"
                            }
      
        # ... = CBasicConvModule(self, oCommonModuleConfig, self.ConvLayerFeatures[3],p_bIsMaxPoolDownsampling=True)

        self.Module1 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[0])
        self.DropoutConv = layers.Dropout(rate = 0.4)                         #Make rate a self.Config.Value["CNN.DropoutProbability"]
        self.Module2 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[1])
        self.Module3 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[2],p_bIsMaxPoolDownsampling=True)
        self.Module4 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[3],p_bIsMaxPoolDownsampling=True)
        self.Module5 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[4])
        ```

    </p>
    </details>

* <details><summary>Learning rate</summary>
    <p>

        ```python
        "Training.LearningRateScheduling": [[20,0.05], [40,0.01]]
        ```

    </p>
    </details>

* ### Improvement! 83.9% accuracy. Still large overfitting.

## Halved all learning rates
* <details><summary>Learning rate</summary>
    <p>

        ```python
        "Training.LearningRate": 0.05
        ,"Training.LearningRateScheduling": [[20,0.01], [40,0.005]]
        ```

    </p>
    </details>

* ### 84.9% accuracy.

## Enabled weight regularization.
* <details><summary>Regularization</summary>
    <p>

        ```python
        "Training.RegularizeL2": True
        ,"Training.WeightDecay": 1e-4
        ```

    </p>
    </details>

* ### 85.9% accuracy. We are getting there.

## Added some more image augmentation by applying a random crop to the flipped images.
* <details><summary>Augmentation</summary>
    <p>

        ```python
        def PreprocessImageAugmentDataset(p_tImageInTS, p_tLabelInTS):
            # Normalizes color component values from `uint8` to `float32`.
            tNormalizedImage = __normalizeImage(p_tImageInTS)
            # Calls the data augmentation function that add new random samples, i.e.augments the dataset. 
            tNewRandomImage = tf.image.random_flip_left_right(tNormalizedImage)

            # Resizes the image to 40x40 by padding with zeroes.
            tNewRandomImage = tf.image.resize_with_pad(tNewRandomImage, 40, 40, method=tf.image.ResizeMethod.BILINEAR, antialias=False)

            # Randomly crops the image back to 32x32.
            tNewRandomImage = tf.image.random_crop(tNewRandomImage, [32, 32, 3])
    
            
            # Target class labels into one-hot encoding
            tTargetOneHot = tf.one_hot(p_tLabelInTS, CONFIG["CNN.Classes"])
            
            return tNewRandomImage, tTargetOneHot
        ```

    </p>
    </details>

* ### 77.8% accuracy. We are not getting there.

## Changed the size of the resized images to 38 in order to maintain more of the class.
* <details><summary>Augmentation</summary>
    <p>

        ```python
        def PreprocessImageAugmentDataset(p_tImageInTS, p_tLabelInTS):
            # Normalizes color component values from `uint8` to `float32`.
            tNormalizedImage = __normalizeImage(p_tImageInTS)
            # Calls the data augmentation function that add new random samples, i.e.augments the dataset. 
            tNewRandomImage = tf.image.random_flip_left_right(tNormalizedImage)

            # Resizes the image to 40x40 by padding with zeroes.
            tNewRandomImage = tf.image.resize_with_pad(tNewRandomImage, 38, 38, method=tf.image.ResizeMethod.BILINEAR, antialias=False)

            # Randomly crops the image back to 32x32.
            tNewRandomImage = tf.image.random_crop(tNewRandomImage, [32, 32, 3])
    
            
            # Target class labels into one-hot encoding
            tTargetOneHot = tf.one_hot(p_tLabelInTS, CONFIG["CNN.Classes"])
            
            return tNewRandomImage, tTargetOneHot
        ```

    </p>
    </details>

* ### 79.9% accuracy. 

## Removed the flipping of the images and kept only the cropping.
* <details><summary>Augmentation</summary>
    <p>

        ```python
        def PreprocessImageAugmentDataset(p_tImageInTS, p_tLabelInTS):
            # Normalizes color component values from `uint8` to `float32`.
            tNormalizedImage = __normalizeImage(p_tImageInTS)
            # Calls the data augmentation function that add new random samples, i.e.augments the dataset. 
            # tNewRandomImage = tf.image.random_flip_left_right(tNormalizedImage)

            # Resizes the image to 38x38 by padding with zeroes.
            tNewRandomImage = tf.image.resize_with_pad(tNormalizedImage, 38, 38, method=tf.image.ResizeMethod.BILINEAR, antialias=False)

            # Randomly crops the image back to 32x32.
            tNewRandomImage = tf.image.random_crop(tNewRandomImage, [32, 32, 3])
            
            # Target class labels into one-hot encoding
            tTargetOneHot = tf.one_hot(p_tLabelInTS, CONFIG["CNN.Classes"])
            
            return tNewRandomImage, tTargetOneHot
        ```

    </p>
    </details>

* ### 81.23% accuracy. 

## Removed cropping and changed the learning rate scheduler.
* <details><summary>Learning rate</summary>
    <p>

        ```python
        ,"Training.LearningRate": 0.1
        ,"Training.LearningRateScheduling": [[10,0.05], [25,0.01], [40,0.005]]
        ```

    </p>
    </details>

* ### 85.81% accuracy. The accuracy seems to plateau after epoch 25. Perhaps there is no difference between lr 0.01 and lr 0.005.

## Reintroduced cropping with the new learning rates. Just in case
* <details><summary>Augmentation</summary>
    <p>

        ```python
       def PreprocessImageAugmentDataset(p_tImageInTS, p_tLabelInTS):
            # Normalizes color component values from `uint8` to `float32`.
            tNormalizedImage = __normalizeImage(p_tImageInTS)
            # Calls the data augmentation function that add new random samples, i.e.augments the dataset. 
            tNewRandomImage = tf.image.random_flip_left_right(tNormalizedImage)

            # Resizes the image to 38x38 by padding with zeroes.
            tNewRandomImage = tf.image.resize_with_pad(tNewRandomImage, 38, 38, method=tf.image.ResizeMethod.BILINEAR, antialias=False)

            # Randomly crops the image back to 32x32.
            tNewRandomImage = tf.image.random_crop(tNewRandomImage, [32, 32, 3])
            
            # Target class labels into one-hot encoding
            tTargetOneHot = tf.one_hot(p_tLabelInTS, CONFIG["CNN.Classes"])
            
            return tNewRandomImage, tTargetOneHot
        ```

    </p>
    </details>

* <details><summary>Learning rate</summary>
    <p>

        ```python
        ,"Training.LearningRate": 0.1
        ,"Training.LearningRateScheduling": [[10,0.05], [25,0.01], [40,0.005]]
        ```

    </p>
    </details>

* ### 80.3% accuracy. 

## Tried a different approach to random cropping. Cropping now comes before flipping.
* <details><summary>Augmentation</summary>
    <p>

        ```python
       def PreprocessImageAugmentDataset(p_tImageInTS, p_tLabelInTS):
            # Normalizes color component values from `uint8` to `float32`.
            tNormalizedImage = __normalizeImage(p_tImageInTS)
            # Calls the data augmentation function that add new random samples, i.e.augments the dataset. 
            

            # Resizes the image to 40x40 by padding with zeroes.
            tNewRandomImage = tf.image.pad_to_bounding_box(tNormalizedImage, 4, 4, 40, 40)

            # Randomly crops the image back to 32x32.
            tNewRandomImage = tf.image.random_crop(tNewRandomImage, [32, 32, 3])

            # Flips the image randomly
            tNewRandomImage = tf.image.random_flip_left_right(tNewRandomImage)
            
            # Target class labels into one-hot encoding
            tTargetOneHot = tf.one_hot(p_tLabelInTS, CONFIG["CNN.Classes"])
            
            return tNewRandomImage, tTargetOneHot
        ```

    </p>
    </details>

* ### 84.1% accuracy. Seems to hit local minima after changing lr to 0.05

## Changed the learning rate scheduler.
* <details><summary>Learning rate</summary>
    <p>

        ```python
         "Training.LearningRate": 0.1
        ,"Training.LearningRateScheduling": [[10,0.05], [40,0.01]]
        ```

    </p>
    </details>

* ### 84.8% accuracy. Plateaus from epoch 30 to 40.

## Standardized the dataset and I skipped the dropout layer.
* <details><summary>Learning rate</summary>
    <p>

        ```python
        mean = np.mean(oDataset.TSSamples, axis=(0,1,2))
        std = np.std(oDataset.TSSamples, axis=(0,1,2))

        # Standardize the datasets with the mean and st dev of the trainig samples
        oDataset.TSSamples = (oDataset.TSSamples - mean) / std
        oDataset.VSSamples = (oDataset.VSSamples - mean) / std
        print(mean)
        print(std)
        ```

    </p>
    </details>

* ### 83.95% accuracy. Plateaus after setting lr to 0.01.

## Brought back the dropout layer. Just in case.
* ### 84.1% accuracy. 

## Removed cropping again but keeping the flipping and the standardization.
* ### 85% accuracy.

## Reused the best performing learning rates so far.
* <details><summary>Learning rate</summary>
    <p>

        ```python
        "Training.LearningRate": 0.05
        ,"Training.LearningRateScheduling": [[20,0.01], [40,0.005]]
        ```

    </p>
    </details>

* ### 86% accuracy.

## Tried the cropping again. Just in case.
* ### 84.6% accuracy. At this point it seems the model can't perform above 86%.

## Tried adding a 6th inception module but it exhausts the collab resources no matter how the parameters are set.

## Went back to the best working architecture. Tried scheduling an increase of the learning rate during training to break out of the local minima.

* <details><summary>Architecture</summary>
    <p>

        ```python
        oCommonModuleConfig={  "Convolution.Features"           : None 
                              ,"Convolution.PaddingSize"        : 1
                              ,"Convolution.WindowSize"         : 3
                              ,"Convolution.Stride"             : 1
                              ,"Convolution.KernelInitializer"  : "glorot_uniform"
                              ,"Convolution.HasBias"            : False
                              ,"Convolution.BiasInitializer"    : None
                              ,"Convolution.RegularizeL2"       : self.Config.Value["Training.RegularizeL2"]
                              ,"Convolution.WeightDecay"        : self.Config.Value["Training.WeightDecay"]
                              ,"ActivationFunction"             : "relu"
                              ,"Normalization"                  : "BatchNormalization"
                            }
      
        # ... = CBasicConvModule(self, oCommonModuleConfig, self.ConvLayerFeatures[3],p_bIsMaxPoolDownsampling=True)

        self.Module1 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[0])
        self.DropoutConv = layers.Dropout(rate = 0.4)                         #Make rate a self.Config.Value["CNN.DropoutProbability"]
        self.Module2 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[1])
        self.Module3 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[2],p_bIsMaxPoolDownsampling=True)
        self.Module4 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[3],p_bIsMaxPoolDownsampling=True)
        self.Module5 = CInceptionModule(self, oCommonModuleConfig, self.ConvLayerFeatures[4])
        ```

    </p>
    </details>

* <details><summary>Learning rate</summary>
    <p>

        ```python
        "Training.LearningRate": 0.05
        ,"Training.LearningRateScheduling": [[20,0.01], [25,0.05], [30,0.01], [40,0.005]]
        ```

    </p>
    </details>

* ### The increase of the learning rate did get the network out of the local minima by dipping the accuracy by 5%. It then recovered back to 85.9%


