#%%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import SGD, RMSprop
from pprint import pprint

from preprocessing import create_generator, create_validation_generator
image_size = 100
classes = 11
print('DONE')
 
#%%
model = Sequential()
model.add(Flatten(input_shape=(image_size, image_size, 3)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(classes, activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(create_generator(image_size, 2),
                    epochs=20,
                    verbose=2,
                    )

#%%
scores = model.evaluate_generator(create_validation_generator(image_size, 1))
pprint('done')


#%%
pprint(model.metrics_names)
pprint(scores)
