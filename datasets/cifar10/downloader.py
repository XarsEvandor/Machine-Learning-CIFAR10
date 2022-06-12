import os
import shutil
import sys
import zipfile
import tarfile
import pickle
from urllib.request import urlretrieve
import numpy as np


# =======================================================================================================================
class CDataSetDownloaderCIFAR10(object):
  DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_sDataFolder="cifar10"):
    self.DataFolder = os.path.join("MLData", p_sDataFolder)
    self.TempFolder = os.path.join(self.DataFolder, "tmp")
  # --------------------------------------------------------------------------------------------------------            
  def _downloadProgressCallBack(self, count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()        
  # --------------------------------------------------------------------------------------------------------
  def __ensureDataSetIsOnDisk(self):
    if not os.path.exists(self.DataFolder):
      os.makedirs(self.DataFolder)
    if not os.path.exists(self.TempFolder):
      os.makedirs(self.TempFolder)


    sSuffix = CDataSetDownloaderCIFAR10.DOWNLOAD_URL.split('/')[-1]
    sArchiveFileName = os.path.join(self.TempFolder, sSuffix)
        
    if not os.path.isfile(sArchiveFileName):
      sFilePath, _ = urlretrieve(url=CDataSetDownloaderCIFAR10.DOWNLOAD_URL, filename=sArchiveFileName, reporthook=self._downloadProgressCallBack)
      print()
      print("Download finished. Extracting files.")

            
    if sArchiveFileName.endswith(".zip"):
      zipfile.ZipFile(file=sArchiveFileName, mode="r").extractall(self.TempFolder)
    elif sArchiveFileName.endswith((".tar.gz", ".tgz")):
      tarfile.open(name=sArchiveFileName, mode="r:gz").extractall(self.TempFolder)
    print("Done.")

    sSourceFolder = os.path.join(self.TempFolder, "cifar-10-batches-py")
    oFileNames = os.listdir(sSourceFolder)
    
    for sFileName in oFileNames:
        shutil.move(os.path.join(sSourceFolder, sFileName), self.DataFolder)

    os.remove(sArchiveFileName)
    os.rmdir(sSourceFolder)
    os.rmdir(self.TempFolder)
  # --------------------------------------------------------------------------------------------------------
  def Download(self):
    if not os.path.isfile(os.path.join(self.DataFolder, "test_batch")):
      self.__ensureDataSetIsOnDisk()
  # --------------------------------------------------------------------------------------------------------
# =======================================================================================================================


if __name__ == "__main__":
  oDataSet = CDataSetDownloaderCIFAR10()
  oDataSet.Download()