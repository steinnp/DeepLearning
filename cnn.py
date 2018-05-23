
#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import he_normal, glorot_normal
from keras.activations import elu
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from pprint import pprint
from plot import plot_loss_accuracy, plot_model_structure

from preprocessing import create_generator, create_validation_generator
image_size = 100
classes = 120
print('DONE')

#%%
def generate_cnn_model(learn_rate=0.01, momentum=0, epochs=5, activation='softplus', batch_size=100, dropout=0.1):
  model = Sequential()
  # input: 300x300 images with 3 channels -> (100, 100, 3) tensors.
  # this applies 32 convolution filters of size 3x3 each.
  model.add(Conv2D(32,
              (3, 3),
              activation=activation,
              input_shape=(image_size, image_size, 3),
              kernel_initializer=he_normal(),
              padding='same'
              )
        )
  model.add(Dropout(0.1))
  model.add(Conv2D(32,
              (3, 3),
              activation=activation,
              input_shape=(image_size, image_size, 3),
              kernel_initializer=he_normal(),
              padding='same'
              )
        )
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
  model.add(Dropout(0.25))

  model.add(Conv2D(64,
              (3, 3),
              activation=activation,
              kernel_initializer=he_normal(),
              #kernel_regularizer=l2(0.01),
              padding='same'
              )
        )
  model.add(Dropout(0.25))
  model.add(Conv2D(64,
              (3, 3),
              activation=activation,
              kernel_initializer=he_normal(),
              #kernel_regularizer=l2(0.01),
              padding='same'
              )
        )
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(1024,
              activation=activation,
              kernel_initializer=he_normal(),
              #kernel_regularizer=l2(0.01)
              )
        )
  model.add(Dropout(0.5))
  model.add(Dense(1024,
              activation=activation,
              kernel_initializer=he_normal(),
              #kernel_regularizer=l2(0.01)
              )
        )
  model.add(Dropout(0.5))
  model.add(Dense(classes,
              activation='softmax',
              kernel_initializer=he_normal(),
              #kernel_regularizer=l2(0.01)
              )
        )
  sgd = SGD(lr=0.01, decay=1e-5, momentum=0.95, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  history = model.fit_generator(create_generator(image_size, batch_size),
                      epochs=epochs,
                      verbose=2,
                      validation_data=create_validation_generator(image_size, batch_size)
                      )
  return model, history
print('Done')
#%%
"""
batch size:  76 , mo:  [4.8256128809877774, 0.010605476279425152]
batch size:  100 , mo:  [4.6729812914461784, 0.037215579259455366]
batch size:  4 , mo:  [4.7790192858942033, 0.012148091014269186]
batch size:  10 , mo:  [4.779029438130606, 0.012148091195289848]
batch size:  24 , mo:  [4.8111025511965622, 0.01214809137631051]
batch size:  48 , mo:  [4.7584040910637704, 0.020053991992594052]
batch_sizes = [76]
models = []
for bat in batch_sizes:
  print('WORKING ON BATCH: ' + str(bat))
  model = generate_cnn_model(batch_size=bat)
  scores = model.evaluate_generator(create_validation_generator(image_size, bat))
  models.append((bat, scores))

#%%
for mod in models:
    print('batch size: ', mod[0], ', mo: ', mod[1])


"""
#%%
"""
Drop out percentage:  0.1 , mo:  [4.3411649791513041, 0.054184340931851573]
Drop out percentage:  0.2 , mo:  [4.4730773361962228, 0.042036249650361408]
Drop out percentage:  0.4 , mo:  [4.5680525596197903, 0.025260315461653887]
dropout_percentage = [0.1, 0.2, 0.4, 0.6, 0.7]
models = []
for drop in dropout_percentage:
  print('WORKING ON DROP: ' + str(drop))
  model = generate_cnn_model(dropout=drop)
  scores = model.evaluate_generator(create_validation_generator(image_size, 100))
  models.append((drop, scores))

#%%
for mod in models:
    print('Drop out percentage: ', mod[0], ', mo: ', mod[1])

"""
#%%
model, history = generate_cnn_model(epochs = 250, batch_size=100)
# plot_loss_accuracy(history, 'cnnAccuracy.png', 'cnnLoss.png')
#adam = Adam()
#%%
plot_model_structure(model, 'model_structure.png')
#plot_loss_accuracy(history, 'cnnAccuracy.png', 'cnnLoss.png')
#print('Done training')
## Try 20 epochs, batch size 32, decay 1e-5, momentum 0.001, lr 0.05, dropout 0.04, 0.04, 0.02
##%%
#scores = model.evaluate_generator(create_validation_generator(image_size, 16))
#pprint('done')
#
#
##%%
#pprint(model.metrics_names)
#pprint(scores)
