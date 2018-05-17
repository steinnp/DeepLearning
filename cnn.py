
#%%
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from pprint import pprint

from preprocessing import create_generator, create_validation_generator
image_size = 100
classes = 11
print('DONE')

#%%

model = Sequential()
# input: 300x300 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(create_generator(image_size, 16),
                    epochs=5,
                    verbose=1,
                    )

#%%
scores = model.evaluate_generator(create_validation_generator(image_size, 16))
pprint('done')


#%%
pprint(model.metrics_names)
pprint(scores)
