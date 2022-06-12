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
from sklearn import metrics

#==============================================================================================================================
class CEvaluator(object):
    #--------------------------------------------------------------------------------------------------------------
    def __init__(self, p_nActualClasses, p_nPredictedClasses):
        self.ActualClasses      = p_nActualClasses
        self.PredictedClasses   = p_nPredictedClasses

        self.ConfusionMatrix = np.asarray( metrics.confusion_matrix(self.ActualClasses, self.PredictedClasses) )
                
        self.Accuracy        = metrics.accuracy_score(self.ActualClasses, self.PredictedClasses)
        self.Precision, self.Recall, self.F1Score, self.Support = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses, average=None)
        self.AveragePrecision, self.AverageRecall, self.AverageF1Score, self.AverageSupport = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses,  average='weighted')
    #--------------------------------------------------------------------------------------------------------------
    def PrintConfusionMatrix(self):
        nSize = len(self.ConfusionMatrix[0])
        print("                    Predicted  ")
        print("               --" + "-"*5*nSize)
        sLabel = "Actual"
        for nIndex,nRow in enumerate(self.ConfusionMatrix):
            print("        %s | %s |" % (sLabel, " ".join(["%4d" % x for x in nRow])))
            if nIndex == 0:
                sLabel = " "*len(sLabel)       
        print("               --" + "-"*5*nSize)
        
    #--------------------------------------------------------------------------------------------------------------
    
#==============================================================================================================================
