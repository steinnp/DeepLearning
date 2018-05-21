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
classes = 120
print('INIT DONE')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#%%
def generate_ffnn_model(learn_rate=0.01, momentum=0, epochs=10, activation='softplus', dropout=0.1, batch_size=4):
    model = Sequential()
    model.add(Flatten(input_shape=(image_size, image_size, 3)))
    model.add(Dense(1024, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(1024, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(classes, activation='softmax', kernel_initializer='he_normal'))

    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit_generator(create_generator(image_size, batch_size),
                        epochs=epochs,
                        verbose=2,
                        # validation_data=create_validation_generator(image_size, batch_size)
                        )
    return model
print('INIT FUNC DONE')

#%%
# Search for good batch size
batch_sizes = [4, 8, 16, 24, 32]
batch_results = []

for bat in batch_sizes:
    print('CHECKING BATCH SIZE: ' + str(bat))
    new_mod = generate_ffnn_model(batch_size=bat, epochs=5)
    scores = new_mod.evaluate_generator(create_validation_generator(image_size, bat))
    batch_results.append((bat, scores))
print('DONE BATCH SERACH')

"""
(4, [4.47029961428953, 0.0323949093713845])
(8, [4.507607497904138, 0.031430775163902816])
(16, [4.546801352841119, 0.03008098727342846])
(24, [4.579649560386225, 0.029695334389225378])
(32, [4.587032192331734, 0.025067489394523718])
"""


#%%
for res in batch_results:
    print(res)

#%%
# Run the model with default settings
mod = generate_ffnn_model()
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
lr_mo_results = []

for lr in learn_rate:
    for mo in momentum:
        new_mod = generate_ffnn_model(learn_rate=lr, momentum=mo)
        scores = new_mod.evaluate_generator(create_validation_generator(image_size, 1))
        lr_mo_results.append((lr, scores))

#%%
for res in lr_mo_results:
    print(res)
