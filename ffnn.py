#%%
import numpy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import SGD, RMSprop
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from pprint import pprint

from preprocessing import create_generator, create_validation_generator
image_size = 100
classes = 11
print('INIT DONE')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#%%
def generate_ffnn_model(learn_rate=0.01, momentum=0, epochs=10, activation='softplus'):
    model = Sequential()
    model.add(Flatten(input_shape=(image_size, image_size, 3)))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(classes, activation=activation))

    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit_generator(create_generator(image_size, 10),
                        epochs=epochs,
                        verbose=2,
                        validation_data=create_validation_generator(image_size, 10)
                        )
    return model
print('INIT FUNC DONE')


#%%
mod = generate_ffnn_model(learn_rate=0.01, momentum=0, epochs=50)
scores = mod.evaluate_generator(create_validation_generator(image_size, 10))
pprint(scores)

## Hyper Search for Activation ##
#%%
results = []
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

for act in activation:
    new_mod = generate_ffnn_model(activation=act)
    scores = new_mod.evaluate_generator(create_validation_generator(image_size, 10))
    results.append((act, scores))

#%%
for res in results:
    print(res)


## Hyper Search for Learning Rate and Momentum ##
#%%
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
models = []

for lr in learn_rate:
    # for mo in momentum:
    new_mod = generate_ffnn_model(learn_rate=lr)
    scores = new_mod.evaluate_generator(create_validation_generator(image_size, 1))
    models.append((lr, scores))

#%%
for mod in models:
    print('lr: ', mod[0], ', mo: ', mod[1])
    print(mod[2])
