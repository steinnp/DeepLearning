# Used to split files into directory into two directories with a 75/25 split ratio
# to create a test and training dataset
#%%
import os
import math
import shutil
#imagePath = './Data/images/Images/{0}/'
#root = './Data/images/Images'
#imagePath = './Data/images/Images/{0}/'
#trainPath = 'G:/Dogs/train/{0}'
#validationPath = 'G:/Dogs/validation/{0}'

root = 'N:/DL_Data/Data/Images'
imagePath = 'N:/DL_Data/Data/Images/{0}/'
trainPath = 'N:/DL_Data/Dogs/train/{0}'
validationPath = 'N:/DL_Data/Dogs/validation/{0}'

imageDirectories = os.listdir(root)

for breedDir in imageDirectories:
  dogFiles = os.listdir(imagePath.format(breedDir))
  numFiles = len(dogFiles)
  numTrain = math.floor(numFiles * 0.75)
  trainDogs = dogFiles[:numTrain]
  valDogs = dogFiles[numTrain:]

  if not os.path.exists(trainPath.format(breedDir)):
    os.makedirs(trainPath.format(breedDir))
  if not os.path.exists(validationPath.format(breedDir)):
    os.makedirs(validationPath.format(breedDir))
  for dogFile in trainDogs:
    shutil.copy2(imagePath.format(breedDir) + dogFile, trainPath.format(breedDir))
  for dogFile in valDogs:
    shutil.copy2(imagePath.format(breedDir) + dogFile, validationPath.format(breedDir))

print('DONE')