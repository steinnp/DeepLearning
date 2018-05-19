
#%%
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
from pprint import pprint
from sklearn.preprocessing import LabelBinarizer
sess = tf.Session()
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy
# annotationPath = './Data/annotation/Annotation'
# imagePath = './Data/images/Images'
annotationPath = 'N:/DL_Data/Data/Annotation'
imagePath = 'N:/DL_Data/Data/Images'
saveTestPath = 'N:/DL_Data/DoggoTest'
savePath = 'N:/DL_Data/DoggoTrain'
# saveTestPath = 'G:/DogsTest'
# savePath = 'G:/Dogs2'
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
print('done')

#%%
def create_generator(size, batch_size):
  return datagen.flow_from_directory(savePath, batch_size=batch_size, target_size=(size, size), class_mode='categorical')

def create_validation_generator(size, batch_size):
  return val_datagen.flow_from_directory(saveTestPath, batch_size=batch_size, target_size=(size, size), class_mode='categorical')

#%%
def preprocess_image(sourcePath, destinationPath, breed, dataGenerator, copies):
  print('Running preprocess_image')
  img = load_img(sourcePath)
  x = img_to_array(img)
  x = tf.image.resize_images(x, (300, 300)).eval(session=sess)
  x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 200, 200)
  # the .flow() command below generates batches of randomly transformed images
  # and saves the results to the `preview/` directory
  i = 0
  for batch in dataGenerator.flow(x, batch_size=5,
                            save_to_dir=destinationPath, save_prefix=breed, save_format='jpeg'):
      if i >= copies:
          break  # otherwise the generator would loop indefinitely
      i += 1

#%%
def generate_training_data():
  print('Running generate_training_data')
  dogbreedDirectories = os.listdir(annotationPath)
  i = 0
  for breedDir in dogbreedDirectories:
    if i > 10:
      break
    print(breedDir)
    dogfile = os.listdir(annotationPath + '/' + breedDir)[0]
    tree = ET.parse(annotationPath + '/' + breedDir + '/' + dogfile)
    root = tree.getroot()
    obj = root.find('object')
    breed = obj.find('name').text
    print(breed)
    destinationPath = savePath + '/' + breed
    if not os.path.exists(destinationPath):
      os.makedirs(destinationPath)
    j = 0
    for imageFile in os.listdir(imagePath + '/' + breedDir):
      if j > 30:
        break
      preprocess_image(imagePath + '/' + breedDir + '/' + imageFile, destinationPath, breed, datagen, 2)
      j += 1 
    i += 1
  print('DONE')
#%%
def generate_testing_data():
  print('Running generate_testing_data')
  dogbreedDirectories = os.listdir(annotationPath)
  i = 0
  for breedDir in dogbreedDirectories:
    if i > 10:
      break
    print(breedDir)
    dogfile = os.listdir(annotationPath + '/' + breedDir)[0]
    tree = ET.parse(annotationPath + '/' + breedDir + '/' + dogfile)
    root = tree.getroot()
    obj = root.find('object')
    breed = obj.find('name').text
    print(breed)
    destinationPath = saveTestPath + '/' + breed
    if not os.path.exists(destinationPath):
      os.makedirs(destinationPath)
    j = 0
    for imageFile in os.listdir(imagePath + '/' + breedDir):
      if j > 70 and j <= 90:
        preprocess_image(imagePath + '/' + breedDir + '/' + imageFile, destinationPath, breed, val_datagen, 1)
      elif j > 90:
        break
      j += 1
    i += 1
  print('DONE')

#%%
def create_one_hot_dict():
  dogbreedDirectories = os.listdir(annotationPath)
  dogbreeds = []
  for breedDir in dogbreedDirectories:
    dogfile = os.listdir(annotationPath + '/' + breedDir)[0]
    tree = ET.parse(annotationPath + '/' + breedDir + '/' + dogfile)
    root = tree.getroot()
    obj = root.find('object')
    breed = obj.find('name').text
    dogbreeds.append(breed)
  encoder = LabelBinarizer()
  oneHotBreeds = encoder.fit_transform(dogbreeds)
  oneHotDict = {}
  for ind in range(len(dogbreeds)):
    oneHotDict[dogbreeds[ind]] = oneHotBreeds[ind]
  pprint(oneHotDict)
  return oneHotDict