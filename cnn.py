
#%%
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import he_normal, glorot_normal
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from pprint import pprint

from preprocessing import create_generator, create_validation_generator
image_size = 192
classes = 11
print('DONE')

#%%

model = Sequential()
# input: 300x300 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64,
                (3, 3),
                activation='relu',
                input_shape=(image_size, image_size, 3),
                kernel_initializer=he_normal(),
                padding='same'
                )
          )
model.add(Conv2D(64,
                (3, 3),
                activation='relu',
                kernel_initializer=he_normal(),
                padding='same'
                )
          )
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32,
                (3, 3),
                activation='relu',
                kernel_initializer=he_normal(),
                kernel_regularizer=l2(0.01),
                padding='same'
                )
          )
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(16,
                (3, 3),
                activation='relu',
                kernel_regularizer=l2(0.01),
                padding='same'
                )
          )
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64,
                activation='relu',
                kernel_initializer=he_normal(),
                kernel_regularizer=l2(0.01)
               )
          )
model.add(Dropout(0.2))
model.add(Dense(classes,
                activation='relu',
                kernel_initializer=he_normal(),
                kernel_regularizer=l2(0.01)
               )
         )

sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)
adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(create_generator(image_size, 32),
                    epochs=5,
                    verbose=2,
                    )
# Try 20 epochs, batch size 32, decay 1e-5, momentum 0.001, lr 0.05, dropout 0.04, 0.04, 0.02
#%%
scores = model.evaluate_generator(create_validation_generator(image_size, 16))
pprint('done')


#%%
pprint(model.metrics_names)
pprint(scores)
