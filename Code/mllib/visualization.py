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


import numpy as np
import matplotlib.pyplot as plt         # use the subpackage (a.k.a. namespace) with the alias "plt"
from matplotlib import colors           

# ====================================================================================================
class CPlot(object):  # class CPlot: object
    # --------------------------------------------------------------------------------------
    # Constructor
    def __init__(self, p_sTitle, p_oSamples, p_oLabels
                 ,p_sLabelDescriptions=["orange tree","olive tree"]
                 ,p_sColors=["darkorange","darkseagreen"] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
                 ,p_sXLabel="Feature 1"
                 ,p_sYLabel="Feature 2"
                 ):
        # ................................................................
        # // Fields \\
        self.Title = p_sTitle
        self.Samples = p_oSamples
        self.Labels = p_oLabels
        self.LabelDescriptions  = p_sLabelDescriptions
        self.Colors             = p_sColors 
        self.XLabel             = p_sXLabel
        self.YLabel             = p_sYLabel
        # ................................................................
    # --------------------------------------------------------------------------------------
    def Show(self, p_bIsMinMaxScaled=False, p_nLineSlope=None, p_nLineIntercept=None):

        # Two dimensional data for the scatter plot
        nXValues = self.Samples[:,0]
        nYValues = self.Samples[:,1]
        nLabels = self.Labels

        
        oColorMap           = colors.ListedColormap(self.Colors)
    
        fig, ax = plt.subplots(figsize=(8,8))
        plt.scatter(nXValues, nYValues, c=nLabels, cmap=oColorMap)
    
        plt.title(self.Title)
        cb = plt.colorbar()
        nLoc = np.arange(0,max(nLabels),max(nLabels)/float(len(self.Colors)))
        cb.set_ticks(nLoc)
        cb.set_ticklabels(self.LabelDescriptions)
        

        if (p_nLineSlope is not None):
            x1 = np.min(nXValues)
            y1 = p_nLineSlope * x1 + p_nLineIntercept;
            x2 = np.max(nXValues)
            y2 = p_nLineSlope * x2 + p_nLineIntercept;
            oPlot1 = ax.plot([x1,x2], [y1,y2], 'r--', label="Decision line")
            oLegend = plt.legend(loc = "upper left", shadow=True, fontsize='x-large')
            oLegend.get_frame().set_facecolor("lightyellow")

        if p_bIsMinMaxScaled:
            ax.set_xlim( (0.0, 1.0) )
            ax.set_ylim( (0.0, 1.0) )

        ax.set_xlabel(self.XLabel)
        ax.set_ylabel(self.YLabel)


        #plt.scatter(oDataset.Samples[:,0], oDataset.Samples[:,1])
                 #, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

        plt.show()
    # --------------------------------------------------------------------------------------
# ====================================================================================================







# =========================================================================================================================
class CHistogramOfClasses(object):  # class CPlot: object
  # --------------------------------------------------------------------------------------
  def __init__(self, p_nData, p_nClasses, p_bIsProbabilities=False):
    self.Data = p_nData
    self.Classes = p_nClasses
    self.IsProbabilities = p_bIsProbabilities
  # --------------------------------------------------------------------------------------
  def Show(self):

    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.hist(self.Data, density=self.IsProbabilities, bins=self.Classes, ec="k") 
    ax.locator_params(axis='x', integer=True)

    if self.IsProbabilities:
      plt.ylabel('Probabilities')
    else:
      plt.ylabel('Counts')
    plt.xlabel('Classes')
    plt.show()
  # --------------------------------------------------------------------------------------
# =========================================================================================================================



# =========================================================================================================================
class CPlotConfusionMatrix(object):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfusionMatrix):
        self.ConfusionMatrix = p_oConfusionMatrix
    # --------------------------------------------------------------------------------------
    def Show(self):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(self.ConfusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(self.ConfusionMatrix.shape[0]):
            for j in range(self.ConfusionMatrix.shape[1]):
                ax.text(x=j, y=i,s=self.ConfusionMatrix[i, j], va='center', ha='center', size='xx-large')
         
        plt.xlabel('Predicted Label', fontsize=18)
        plt.ylabel('Actual Label'   , fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
    # --------------------------------------------------------------------------------------
# =========================================================================================================================    
    

