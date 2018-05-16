
#%%
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
from pprint import pprint
from sklearn.preprocessing import LabelBinarizer
import numpy
print('done')
annotationPath = './Data/annotation/Annotation'
imagePath = './Data/images/Images'
savePath = 'G:/Dogs'

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

create_one_hot_dict()

def resize(image):
  return tf.image.resize_image_with_crop_or_pad(numpy.asarray(image), 200, 200)

#%%
sess = tf.Session()
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
def preprocess_image(sourcePath, destinationPath, breed):
  datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

  img = load_img(sourcePath)
  x = img_to_array(img)
  x = tf.image.resize_image_with_crop_or_pad(x, 300, 300).eval(session=sess)
  x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 200, 200)
  if not os.path.exists(destinationPath):
    os.makedirs(destinationPath)
  # the .flow() command below generates batches of randomly transformed images
  # and saves the results to the `preview/` directory
  i = 0
  for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=destinationPath, save_prefix=breed, save_format='jpeg'):
      i += 1
      if i > 10:
          break  # otherwise the generator would loop indefinitely

print('DONE')

# preprocess_image('./Data/images/Images/n02085620-Chihuahua/n02085620_199.jpg', 'G:/Dogs/chihuaua', 'chihuaua')          

#%%
def generate_training_data():
  dogbreedDirectories = os.listdir(annotationPath)
  for breedDir in dogbreedDirectories:
    print(breedDir)
    dogfile = os.listdir(annotationPath + '/' + breedDir)[0]
    tree = ET.parse(annotationPath + '/' + breedDir + '/' + dogfile)
    root = tree.getroot()
    obj = root.find('object')
    breed = obj.find('name').text
    print(breed)
    destinationPath = savePath + '/' + breed
    for imageFile in os.listdir(imagePath + '/' + breedDir):
      preprocess_image(imagePath + '/' + breedDir + '/' + imageFile, destinationPath, breed)
  print('DONE')

generate_training_data()
