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


import os
import json
import pickle

# =======================================================================================================================
class CFileStore(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_sBaseFolder, p_bIsVerbose=False):
    #.......................... |  Instance Attributes | ............................
    self.BaseFolder = p_sBaseFolder
    self.IsVerbose  = p_bIsVerbose
    #................................................................................
    if not os.path.exists(self.BaseFolder):
      os.makedirs(self.BaseFolder)
  # --------------------------------------------------------------------------------------------------------
  @property
  def HasData(self):
    bResult = os.path.exists(self.BaseFolder)
    if bResult:
      oFiles = os.listdir(self.BaseFolder)
      nFileCount = len(oFiles)
      bResult = nFileCount > 0

    return bResult;
  # --------------------------------------------------------------------------------------------------------
  def Deserialize(self, p_sFileName, p_bIsPython2Format=False):
    """
    Deserializes the data from a pickle file if it exists.
    Parameters
        p_sFileName        : Full path to the  python object file 
    Returns
        The object with its data or None when the file is not found.
    """
    oData=None
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
       
    if os.path.isfile(p_sFileName):
      if self.IsVerbose :
        print("      {.} Loading data from %s" % p_sFileName)

      with open(p_sFileName, "rb") as oFile:
        if p_bIsPython2Format:
          oUnpickler = pickle._Unpickler(oFile)
          oUnpickler.encoding = 'latin1'
          oData = oUnpickler.load()
        else:
          oData = pickle.load(oFile)
        oFile.close()
        
    return oData
  #----------------------------------------------------------------------------------
  def WriteTextToFile(self, p_sFileName, p_sText):
    """
    Writes text to a file

    Parameters
        p_sFileName        : Full path to the text file
        p_sText            : Text to write
    """
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
    
    if self.IsVerbose :
      print("  {.} Saving text to %s" % p_sFileName)

    with open(p_sFileName, "w") as oFile:
      print(p_sText, file=oFile)
      oFile.close()

    return True
  #----------------------------------------------------------------------------------
  def Serialize(self, p_sFileName, p_oData, p_bIsOverwritting=False, p_sExtraDisplayLabel=None):
    """
    Serializes the data to a pickle file if it does not exists.
    Parameters
        p_sFileName        : Full path to the  python object file 
    Returns
        True if a new file was created
    """
    bResult=False
    
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)

    if p_bIsOverwritting:
      bMustContinue = True
    else:
      bMustContinue = not os.path.isfile(p_sFileName)
        
    if bMustContinue:
      if self.IsVerbose :
        if p_sExtraDisplayLabel is not None:
            print("  {%s} Saving data to %s" % (p_sExtraDisplayLabel, p_sFileName) )                    
        else:
            print("  {.} Saving data to %s" % p_sFileName)
      with open(p_sFileName, "wb") as oFile:
          pickle.dump(p_oData, oFile, pickle.HIGHEST_PROTOCOL)
          oFile.close()
      bResult=True
    else:
      if self.IsVerbose:
          if p_sExtraDisplayLabel is not None:
              print("  {%s} Not overwritting %s" % (p_sExtraDisplayLabel, p_sFileName) )                    
          else:
              print("  {.} Not overwritting %s" % p_sFileName)
                            
    return bResult
  #----------------------------------------------------------------------------------
# =======================================================================================================================