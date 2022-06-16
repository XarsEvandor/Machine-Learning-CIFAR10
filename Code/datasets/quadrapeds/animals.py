##/*------------------------------------------------------------------11/Nov/87-*
##/*  MAKEANIMAL
##/*      Creates an entire trial size of tables. This is a meta program
##/*  on top of the old make*.c files. Works by choosing a random # from 
##/*  one to four to choose which template to use.
##/* 
##/*      This set of animals is carefully designed so that there are some 
##/*  "good" attributes to leave out. Note that it's also for Classit-3 and
##/*   up.
##/*---------------------------------------------------------------------JHG---*/

#
#  Ported to Python 10/02/2021 by Pantelis I. Kaplanoglou for the needs of CS345
#
import numpy as np
import sys
import os
import scipy.stats as stats
from random import randint
from mllib.utils import RandomSeed
from mllib.log import CScreenLog 
from mllib.filestore import CFileStore
from mllib.data import CCustomDataSet
from mllib.visualization import CHistogramOfClasses

# =========================================================================================================================
class CQuadrapedsDataSet(CCustomDataSet):
  # Constructor
  def __init__(self, p_nSampleCount, p_nRandomSeed=2022, p_bIsVerbose=False):
    super(CQuadrapedsDataSet, self).__init__(p_nSampleCount=p_nSampleCount, p_nRandomSeed=p_nRandomSeed)
    # ................................................................
    # // Fields \\
    self.ClassCount         = 4
    self.ClassNames         = ["Dog", "Cat", "Horse", "Giraffe"]
    self.FeatureCount       = 72
    self.Samples            = None
    self.Labels             = None
    
    self.featureGroupIndex  = 0 # Refactoring the printpart function from the 80's to self.AssignPartFeatures 
    self.currentSampleIndex = 0

    self.Log = CScreenLog()
    self.Log.IsShowing = p_bIsVerbose
    self.FileStore = CFileStore("MLData")
    # ................................................................

    # Lazy dataset initialization. Try to load the data and if not already cached to local filestore, generate the samples now and cache them.
    self.Samples = self.FileStore.Deserialize("Quadrapeds1k-Samples.pkl")
    self.Labels  = self.FileStore.Deserialize("Quadrapeds1k-Labels.pkl")

    if self.Samples is None:
      self.Samples            = np.zeros((self.SampleCount, self.FeatureCount), dtype=np.float32)
      self.Labels             = np.zeros((self.SampleCount), dtype=np.int)
      self.GenerateData()

      self.FileStore.Serialize("Quadrapeds1k-Samples.pkl", self.Samples)
      self.FileStore.Serialize("Quadrapeds1k-Labels.pkl", self.Labels)

  # --------------------------------------------------------------------------------------
  def GenerateData(self):
    self.Log.Print("%% make animal run for %d objects, starting with self.RandomSeed %d\n%%\n" % (self.SampleCount, self.RandomSeed))

    self.MakeUniformRandomClasses() # This will generate an almost equal percentage of animals in the dataset
    #self.MakeImbalancedGaussianRandomClasses() # This will generate an imbalanced dataset with more Giraffes than other animals and fewer Horses.

    for nIndex in range(0, self.SampleCount):
      self.featureGroupIndex = 0
      self.currentSampleIndex = nIndex
      nClassLabel = self.Labels[self.currentSampleIndex]
      
      if nClassLabel == 0:
        self.makedog("D")
      elif nClassLabel == 1:
        self.makecat("C")
      elif nClassLabel == 2:
        self.makehorse("H")
      elif nClassLabel == 3:
        self.makegee("G")
      else:
        self.Log.Print("what the hell???\n")
      if ((nIndex %  100) == 0):
        print("Generating sample #%d" % nIndex)

    if ((nIndex % 100) != 0):
      print("Generating sample #%d" % nIndex)
  # --------------------------------------------------------------------------------------
  def MakeUniformRandomClasses(self):
    for nIndex in range(0, self.SampleCount):
      nClassNumber = randint(1,4)
      self.Labels[nIndex] = nClassNumber -1 
  # --------------------------------------------------------------------------------------
  def MakeImbalancedGaussianRandomClasses(self, p_nSigma=0.5, p_nScale=3):
    # Discrete distribution that approximates a gaussian distribution.
    x = np.arange(0, 4)
    xU, xL = x + p_nSigma, x - p_nSigma
    prob = stats.norm.cdf(xU, scale = p_nScale) - stats.norm.cdf(xL, scale = p_nScale)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    self.Labels = np.random.choice(x, size = self.SampleCount, p = prob).astype(np.int)
  # --------------------------------------------------------------------------------------
  ##/*-----------------------------------------------------------------12/Aug/86-*/
  ##/*  
  ##/*	MAKEDOGGIE - a routine to generate a dog, as described by a set
  ##/*		of cylinder and WMS-like attributes. This description is 
  ##/*		self.Log.Printed onto standard out.
  ##/*	Currently, all dogs created herein have the same orientation.
  ##/* 
  ##/*	Object numbers are not assigned by this program, nor is the object
  ##/*	tree created, nor is the object normalized by overall size and mass.
  ##/*	However, the location and axis of sub parts ARE object-centered.
  ##/* 
  ##/*--------------------------------------------------------------------JHG----*/

  def makedog(self, name):
  #char *name;
    #float	self.self.bellrand();
    #float	height, radius, axis[3], location[3];
    axis      = np.zeros(3, dtype=np.float32)
    location  = np.zeros(3, dtype=np.float32)
    #int		texture, fd;
    #float	leglength, torsoH, torsoR;

    self.Log.Print("%%\t\t\t------------ A Dog ------------\n");
    self.Log.Print("%% created with %d\n%%\n" % self.RandomSeed);
    self.Log.Print("%s\tname\n", name);
    self.Log.Print("8\tNcomponents\n");     #/* For Now.... */

    self.Log.Print("%%    TORSO\n");
    self.Log.Print("9		Natts\n");
    location[0]= location[1]= location[2]= 0;
    axis[0]= 1; axis[1]= 0; axis[2]= 0;
    torsoH= height= self.bellrand(22.0, 28.0);
    torsoR= radius= self.bellrand(5.5, 7.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG1\n");
    self.Log.Print("9		Natts\n");
    leglength= self.bellrand(10.0, 15.0);
    location[0]= torsoH/2; 
    location[1]= torsoR; 
    location[2]= -torsoR - leglength/2;
        #/* this leg might be canted forward up to 45 degrees: */
    axis[0]= self.bellrand(0.0,1.0); 
    axis[1]= 0; 
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(0.8, 1.2);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG2\n");
    self.Log.Print("9		Natts\n");
    location[0]= torsoH/2;
    location[1]= -torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= 0;
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(0.8, 1.2);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG3\n");
    self.Log.Print("9		Natts\n");
    location[0]= -torsoH/2;
    location[1]= torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= 0;
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(0.8, 1.2);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG4\n");
    self.Log.Print("9		Natts\n");
    location[0]= -torsoH/2;
    location[1]= -torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= self.bellrand(0.0,1.0);
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(0.8, 1.2);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    NECK\n");
    self.Log.Print("9		Natts\n");
    height= self.bellrand(5.0, 6.0);
    location[0]= torsoH/2 + height/2;
    location[1]= 0;
    location[2]= torsoR;
    axis[0]= 1;
    axis[1]= 0;
    axis[2]= 1;
    radius= self.bellrand(2.0, 3.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    HEAD\n");
    self.Log.Print("9		Natts\n");
    location[0]= torsoH/2 + height; #/* height is neck length */
    location[1]= 0;
    location[2]= torsoR;
    axis[0]= 1;
    axis[1]= self.bellrand(-1.0, 1.0);	#/* he's looking to the right or left */
    axis[2]= 0;
    height= self.bellrand(10.0, 13.0);
    radius= self.bellrand(2.5, 3.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    TAIL\n");
    self.Log.Print("9		Natts\n");
    height= self.bellrand(6.0, 8.0);    #/* a short tailed dog */
    location[0]= -torsoH + height;
    location[1]= 0;
    location[2]= 0;
    axis[0]= -1.0;
    axis[1]= self.bellrand(-1.0,1.0);   #/* pointing any which way */
    axis[2]= self.bellrand(-1.0,1.0);
    radius= self.bellrand(0.2, 0.4);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

  # --------------------------------------------------------------------------------------
  ##/*----------------------------------------------------------------12/Aug/86-*/
  ##/*	MAKEKITTY  - a routine to generate a cat, as described by a set
  ##/*		of cylinder and WMS-like attributes. This description is 
  ##/*		self.Log.Printed onto standard out.
  ##/* 
  ##/*-------------------------------------------------------------------JHG----*/

  def makecat(self, name):
  #char *name;
      #float	self.bellrand();
      #float	height, radius, axis[3], location[3];
      axis      = np.zeros(3, dtype=np.float32)
      location  = np.zeros(3, dtype=np.float32)

      #int		texture;
      #float	leglength, torsoH, torsoR;

      self.Log.Print("%%\t\t\t------------ A Cat ------------\n");
      self.Log.Print("%% created with %d\n%%\n" % self.RandomSeed);
      self.Log.Print("%s\tname\n" % name);
      self.Log.Print("8\tNcomponents\n");     #/* For Now.... */

      self.Log.Print("%%    TORSO\n");
      self.Log.Print("9		Natts\n");
      location[0]= location[1]= location[2]= 0;
      axis[0]= 1; axis[1]= 0; axis[2]= 0;
      torsoH= height= self.bellrand(15.0, 22.0);
      torsoR= radius= self.bellrand(2.5, 4.5);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    LEG1\n");
      self.Log.Print("9		Natts\n");
      leglength= self.bellrand(4.0, 9.0);
      location[0]= torsoH/2; 
      location[1]= torsoR; 
      location[2]= -torsoR - leglength/2;
         #/* this leg might be canted forward up to 45 degrees: */
      axis[0]= self.bellrand(0.0,1.0); 
      axis[1]= 0; 
      axis[2]= -1;
      height= leglength;
      radius= self.bellrand(0.4, 0.8);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    LEG2\n");
      self.Log.Print("9		Natts\n");
      location[0]= torsoH/2;
      location[1]= -torsoR;
      location[2]= -torsoR - leglength/2;
      axis[0]= 0;
      axis[1]= 0;
      axis[2]= -1;
      height= leglength;
      radius= self.bellrand(0.4, 0.8);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    LEG3\n");
      self.Log.Print("9		Natts\n");
      location[0]= -torsoH/2;
      location[1]= torsoR;
      location[2]= -torsoR - leglength/2;
      axis[0]= 0;
      axis[1]= 0;
      axis[2]= -1;
      height= leglength;
      radius= self.bellrand(0.4, 0.8);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    LEG4\n");
      self.Log.Print("9		Natts\n");
      location[0]= -torsoH/2;
      location[1]= -torsoR;
      location[2]= -torsoR - leglength/2;
      axis[0]= self.bellrand(0.0,1.0);
      axis[1]= 0;
      axis[2]= -1;
      height= leglength;
      radius= self.bellrand(0.4, 0.8);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    NECK\n");
      self.Log.Print("9		Natts\n");
      height= self.bellrand(2.0, 4.0);
      location[0]= torsoH/2 + height/2;
      location[1]= 0;
      location[2]= torsoR;
      axis[0]= 1;
      axis[1]= 0;
      axis[2]= 1;
      radius= self.bellrand(1.5, 2.5);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    HEAD\n");
      self.Log.Print("9		Natts\n");
      location[0]= torsoH/2 + height; #/* height is neck length */
      location[1]= 0;
      location[2]= torsoR;
      axis[0]= 1;
      axis[1]= self.bellrand(-1.0, 1.0);	#/* he's looking to the right or left */
      axis[2]= 0;
      height= self.bellrand(3.0, 5.0);
      radius= self.bellrand(1.5, 2.5);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);

      self.Log.Print("%%    TAIL\n");
      self.Log.Print("9		Natts\n");
      height= self.bellrand(10.0, 18.0);    #/* a long tailed cat */
      location[0]= -torsoH + height;
      location[1]= 0;
      location[2]= 0;
      axis[0]= -1.0;
      axis[1]= self.bellrand(-1.0,1.0);   #/* pointing any which way */
      axis[2]= self.bellrand(-1.0,1.0);
      radius= self.bellrand(0.3, 0.7);
      texture= self.bellrand(170.0, 180.0);
      self.AssignPartFeatures(height, radius, texture, axis, location);
  # --------------------------------------------------------------------------------------
  ##/*------------------------------------------------------------------04/Nov/86-*
  ##/*  
  ##/*	MAKEHORSEY - a routine to generate a horse, as described by a set
  ##/*		of cylinder and WMS-like attributes. This description is 
  ##/*		self.Log.Printed onto standard out.
  ##/*	Currently, all horses created herein have the same orientation.
  ##/* 
  ##/*	Object numbers are not assigned by this program, nor is the object
  ##/*	tree created, nor is the object normalized by overall size and mass.
  ##/*	However, the location and axis of sub parts ARE object-centered.
  ##/* 
  ##/*-------------------------------------------------------------------JHG----*/

  def makehorse(self, name):
  #char *name;

    #float	self.bellrand();
    #float	height, radius, axis[3], location[3];
    axis      = np.zeros(3, dtype=np.float32)
    location  = np.zeros(3, dtype=np.float32)
    #int		texture;
    #float	leglength, torsoH, torsoR;

    self.Log.Print("%%\t\t\t------------ A Horse ------------\n");
    self.Log.Print("%% created with %d\n%%\n" % self.RandomSeed);
    self.Log.Print("%s\tname\n" % name);
    self.Log.Print("8\tNcomponents\n");     #/* For Now.... */

    self.Log.Print("%%    TORSO\n");
    self.Log.Print("9		Natts\n");
    location[0]= location[1]= location[2]= 0;
    axis[0]= 1; axis[1]= 0; axis[2]= 0;
    torsoH= height= self.bellrand(50.0, 60.0);
    torsoR= radius= self.bellrand(10.0, 14.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG1\n");
    self.Log.Print("9		Natts\n");
    leglength= self.bellrand(36.0, 44.0);
    location[0]= torsoH/2; 
    location[1]= torsoR; 
    location[2]= -torsoR - leglength/2;
        #/* this leg might be canted forward up to 30 degrees: */
    axis[0]= self.bellrand(0.0,0.5); 
    axis[1]= 0; 
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 3.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG2\n");
    self.Log.Print("9		Natts\n");
    location[0]= torsoH/2;
    location[1]= -torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= 0;
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 3.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG3\n");
    self.Log.Print("9		Natts\n");
    location[0]= -torsoH/2;
    location[1]= torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= 0;
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 3.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG4\n");
    self.Log.Print("9		Natts\n");
    location[0]= -torsoH/2;
    location[1]= -torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= self.bellrand(0.0,0.5);
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 3.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    NECK\n");
    self.Log.Print("9		Natts\n");
    height= self.bellrand(12.0, 16.0);
    location[0]= torsoH/2 + height/2.828;
    location[1]= 0;
    location[2]= torsoR + height/2.828;
    axis[0]= 1;
    axis[1]= 0;
    axis[2]= 1;
    radius= self.bellrand(5.0, 7.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    HEAD\n");
    self.Log.Print("9		Natts\n");
    location[0]= torsoH/2 + height/1.414; #/* height is neck length */
    location[1]= 0;
    location[2]= torsoR + height/1.414;
    axis[0]= 1;
    axis[1]= self.bellrand(-1.0, 1.0);	#/* he's looking to the right or left */
    axis[2]= 0;
    height= self.bellrand(18.0, 22.0);
    radius= self.bellrand(4.0, 6.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    TAIL\n");
    self.Log.Print("9		Natts\n");
    height= self.bellrand(26.0, 33.0);    #/* a long tailed horse */
    location[0]= -torsoH + height;
    location[1]= 0;
    location[2]= 0;
    axis[0]= -1.0;
    axis[1]= self.bellrand(-1.0,1.0);   #/* pointing any which way */
    axis[2]= self.bellrand(-1.0,0.0);
    radius= self.bellrand(1.0, 2.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);


  # --------------------------------------------------------------------------------------
  ##/*----------------------------------------------------------------12/Aug/86-*/
  ##/*  
  ##/*	MAKEGIRAFFE - a routine to generate a giraffe, as described by a set
  ##/*		of cylinder and WMS-like attributes. This description is 
  ##/*		self.Log.Printed onto standard out.
  ##/*	Currently, all giraffes created herein have the same orientation.
  ##/* 
  ##/*	Object numbers are not assigned by this program, nor is the object
  ##/*	tree created, nor is the object normalized by overall size and mass.
  ##/*	However, the location and axis of sub parts ARE object-centered.
  ##/* 
  ##/*-------------------------------------------------------------------JHG----*/

  def makegee(self, name):
  #char *name;
    #float	self.bellrand();
    axis      = np.zeros(3, dtype=np.float32)
    location  = np.zeros(3, dtype=np.float32)
    #float	height, radius, axis[3], location[3];
    #int		texture;
    #float	leglength, torsoH, torsoR;


    self.Log.Print("%%\t\t\t------------ A Giraffe ------------\n");
    self.Log.Print("%% created with %d\n%%\n" % self.RandomSeed);
    self.Log.Print("%s\tname\n" % name);
    self.Log.Print("8\tNcomponents\n");     #/* For Now.... */

    self.Log.Print("%%    TORSO\n");
    self.Log.Print("9		Natts\n");
    location[0]= location[1]= location[2]= 0;
    axis[0]= 1; axis[1]= 0; axis[2]= 0;
    torsoH= height= self.bellrand(60.0, 72.0);
    torsoR= radius= self.bellrand(12.5, 17.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG1\n");
    self.Log.Print("9		Natts\n");
    leglength= self.bellrand(58.0, 70.0);
    location[0]= torsoH/2; 
    location[1]= torsoR; 
    location[2]= -torsoR - leglength/2;
        #/* this leg might be canted forward up to 30 degrees: */
    axis[0]= self.bellrand(0.0,0.5); 
    axis[1]= 0; 
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 4.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG2\n");
    self.Log.Print("9		Natts\n");
    location[0]= torsoH/2;
    location[1]= -torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= 0;
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 4.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG3\n");
    self.Log.Print("9		Natts\n");
    location[0]= -torsoH/2;
    location[1]= torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= 0;
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 4.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    LEG4\n");
    self.Log.Print("9		Natts\n");
    location[0]= -torsoH/2;
    location[1]= -torsoR;
    location[2]= -torsoR - leglength/2;
    axis[0]= self.bellrand(0.0,0.5);
    axis[1]= 0;
    axis[2]= -1;
    height= leglength;
    radius= self.bellrand(2.0, 4.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    NECK\n");
    self.Log.Print("9		Natts\n");
    height= self.bellrand(45.0, 55.0);
    location[0]= torsoH/2 + height/2.828;
    location[1]= 0;
    location[2]= torsoR + height/2.828;
    axis[0]= 1;
    axis[1]= 0;
    axis[2]= 1;
    radius= self.bellrand(5.0, 9.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    HEAD\n");
    self.Log.Print("9		Natts\n");
    location[0]= torsoH/2 + height/1.414; #/* height is neck length */
    location[1]= 0;
    location[2]= torsoR + height/1.414;
    axis[0]= 1;
    axis[1]= self.bellrand(-1.0, 1.0);	#/* he's looking to the right or left */
    axis[2]= 0;
    height= self.bellrand(18.0, 22.0);
    radius= self.bellrand(3.5, 5.5);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

    self.Log.Print("%%    TAIL\n");
    self.Log.Print("9		Natts\n");
    height= self.bellrand(20.0, 25.0);    #/* a short tailed giraffe */
    location[0]= -torsoH + height;
    location[1]= 0;
    location[2]= 0;
    axis[0]= -1.0;
    axis[1]= self.bellrand(-1.0,1.0);   #/* pointing any which way */
    axis[2]= self.bellrand(-1.0,0.0);
    radius= self.bellrand(0.5, 1.0);
    texture= self.bellrand(170.0, 180.0);
    self.AssignPartFeatures(height, radius, texture, axis, location);

  # --------------------------------------------------------------------------------------
  ##/*-----------------------------------------------------------------12/Aug/86-*/
  ##/*  Procedure  - self.AssignPartFeatures
  ##/* 
  ##/*  Inputs    -> height, radius, texture, axis, location.
  ##/* 
  ##/*  Actions   -> self.Log.Prints these values to stnd out. 
  ##/*--------------------------------------------------------------------JHG----*/

  def AssignPartFeatures(self, p_nHeight, p_nRadius, p_nTexture, p_nAxis, p_nLocation, p_bIsVerbose=False):
  #float h,r, axis[], loc[];
  #int t;
    #double sqrt();
    
    if p_bIsVerbose:
      self.Log.Print("%5.2f\tlocX\n %5.2f\tlocY\n %5.2f\tlocZ\n" % (p_nLocation[0],p_nLocation[1],p_nLocation[2]))
      self.Log.Print("%5.2f\taxisX\n %5.2f\taxisY\n %5.2f\taxisZ\n" % (p_nAxis[0], p_nAxis[1], p_nAxis[2]))
      self.Log.Print("%5.2f\t height\n" % p_nHeight);
      self.Log.Print("%5.2f\t radius\n" % p_nRadius);
      self.Log.Print("%d\t texture\n%%\n" % p_nTexture);

    nBaseIndex = self.featureGroupIndex * 9

    self.Samples[self.currentSampleIndex, nBaseIndex:nBaseIndex + 3]      = p_nLocation[:]
    self.Samples[self.currentSampleIndex, nBaseIndex + 3:nBaseIndex + 6]  = p_nAxis[:]
    self.Samples[self.currentSampleIndex, nBaseIndex + 6]                 = p_nHeight
    self.Samples[self.currentSampleIndex, nBaseIndex + 7]                 = p_nRadius
    self.Samples[self.currentSampleIndex, nBaseIndex + 8]                 = p_nTexture

    self.featureGroupIndex += 1

  # --------------------------------------------------------------------------------------
  ##/*-----------------------------------------------------------------10/Jan/87-*/
  ##/*  Function  - self.bellrand
  ##/* 
  ##/*  Inputs    -> st, fin: the range of the random number. Also uses the 
  ##/*	global "self.RandomSeed".
  ##/* 
  ##/*  Returns   ->  A crude approximation of a bell shaped distribution over
  ##/*	the specified range. Values at the center of the range are 3 times
  ##/*	more likely to occur than those at the edges.
  ##/*-------------------------------------------------------------------JHG----*/

  def bellrand(self, st, fin):
  #float st, fin;
	  #float interval;
	  #float random();
    interval= (fin-st)/5.0
    nCenter = np.abs(st - fin) / 2.0
    nResult = np.random.normal(nCenter, interval)
    return nResult
	  #switch (rand3()) {
	  #  case 1: return(random(st, fin));
	  #  case 2: return(random(st+interval, fin-interval));
	  #  case 3: return(random(st+2*interval, fin-2*interval));
	  #  default: self.Log.Print("what the hell???\n"); exit(0);

  # --------------------------------------------------------------------------------------
  ##/*------------------------------------------------------------------13/Feb/86-*/
  ##/*  Function  - RANDOM
  ##/* 
  ##/*  Inputs    -> st, fin: these define the range of the random number.
  ##/*		NOTE: this also needs a global called "self.RandomSeed".
  ##/* 
  ##/*  Returns   -> a (pseudo) random number. (of type float)
  ##/*---------------------------------------------------------------------JHG----*/

  #MOD = 65537
  #MULT = 25173
  #INC = 13849

  #def	ABS(x):
  # return ((x>0) ? x : -x)

  #def random(st,fin):
  ##float st, fin;
  #	float  range;
  #	float  tmp;
  #	range = fin - st;

  #        #/*  "%" is the modulus operator */

  #	self.RandomSeed = (MULT * self.RandomSeed + INC) % MOD;  
  #	tmp = ABS( self.RandomSeed / (float)MOD ) ;
  #	return(tmp * range + st);


  ##/***************************************************************************/
  ##/*  A local function which returns a random integer between 1 and 4.  
  ##/***************************************************************************/

  #def rand3(self):
	 # #self.RandomSeed= (MULT *self.RandomSeed + INC) %MOD;
	 # #return ( ABS(self.RandomSeed/ (float)MOD ) * 3.0 + 1.0)

  #def rand4(self):
  #  #self.RandomSeed= (MULT *self.RandomSeed + INC) %MOD;
  #  #return ( ABS(self.RandomSeed/ (float)MOD ) * 4.0 + 1.0)




if __name__ == "__main__":
  if (len(sys.argv)==3):
    print("usage: '%s <self.RandomSeed> <# of objects>'\n", sys.argv[0])
    oDataSet = CQuadrapedsDataSet(int(sys.argv[1]), int(sys.argv[2]))
    exit(0)
  elif (len(sys.argv) == 1):
    oDataSet = CQuadrapedsDataSet(1000)
  else:
    print("usage: '%s <self.RandomSeed> <# of objects>'\n", sys.argv[0])
    exit(0)


  print("Dataset samples shape: %s" % str(oDataSet.Samples.shape))
  print("Dataset labels shape: %s" % str(oDataSet.Labels.shape))
  oHistogram = CHistogramOfClasses(oDataSet.Labels, oDataSet.ClassCount)
  oHistogram.Show()





