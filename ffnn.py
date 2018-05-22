#%%
import numpy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import SGD, RMSprop
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from pprint import pprint
from plot import plot_loss_accuracy

from preprocessing import create_generator, create_validation_generator
image_size = 100
classes = 120
print('INIT DONE')

# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)

#%%
def generate_ffnn_model(learn_rate=0.01, momentum=0, epochs=10, activation='softplus', dropout=0.1, batch_size=128, optimizer='Adam'):
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

    # sgd = SGD(lr=learn_rate, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit_generator(create_generator(image_size, batch_size),
                        epochs=epochs,
                        verbose=2,
                        validation_data=create_validation_generator(image_size, batch_size)
                        )
    return model, history
print('INIT FUNC DONE')


#%%
# Run the model with default settings
mod, hist = generate_ffnn_model(batch_size=128, epochs=200)
scores = mod.evaluate_generator(create_validation_generator(image_size, 128))
#plot_loss_accuracy(hist, 'ffnn_acc.png', 'ffnn_loss.png')
pprint(scores)
# 
# #%%
# # Search for good batch size
# batch_sizes = [256, 128, 64]
# batch_results = []
# 
# for bat in batch_sizes:
#     print('CHECKING BATCH SIZE: ' + str(bat))
#     new_mod, hist = generate_ffnn_model(batch_size=bat, epochs=5)
#     scores = new_mod.evaluate_generator(create_validation_generator(image_size, bat))
#     batch_results.append((bat, scores))
# print('DONE BATCH SERACH')
# 
# #%%
# for res in batch_results:
#     print(res)
# """
# ADAM
# (256, [15.972018998054478, 0.009062861556074494])
# (128, [4.5469816039071045, 0.026802930973737434])
# (64, [15.94715523857708, 0.010605476282298496])
# 
# SGD
# (4, [4.47029961428953, 0.0323949093713845])
# (8, [4.507607497904138, 0.031430775163902816])
# (16, [4.546801352841119, 0.03008098727342846])
# (24, [4.579649560386225, 0.029695334389225378])
# (32, [4.587032192331734, 0.025067489394523718])
# """
# 
# #%%
# optimizers = ['Adagrad', 'Adadelta', 'Adam', 'SGD', 'Adamax', 'Nadam']
# optimizer_results = [] 
# for op in optimizers:
#     print('CHECKING OPTIMIZERS: ' + op)
#     new_mod, hist = generate_ffnn_model(optimizer=op, epochs=5)
#     scores = new_mod.evaluate_generator(create_validation_generator(image_size, 128))
#     optimizer_results.append((op, scores))
# print('DONE OPTIMIZER SERACH')
# 
# """
# ALL WITH BATCH SIZE 4
# ('Adagrad', [15.993775340888291, 0.007713073659853451])
# ('Adadelta', [15.984451336608721, 0.008291554184342461])
# ('Adam', [15.999991343741337, 0.007327419976860779])
# ('SGD', [4.454804638569483, 0.03740840725028924])
# ('Adamax', [15.993775340888291, 0.007713073659853451])
# ('Nadam', [15.975127332329153, 0.00887003470883147])
# """
# 
# 
# 
# #%%
# for res in optimizer_results:
#     print(res)
# 
# ## Hyper Search for Activation ##
# #%%
# results = []
# activation = ['softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# 
# for act in activation:
#     new_mod, hist = generate_ffnn_model(activation=act)
#     scores = new_mod.evaluate_generator(create_validation_generator(image_size, 4))
#     results.append((act, scores))
# 
# #%%
# for res in results:
#     print(res)
# 
# """
# ('softplus', [4.423347639200819, 0.031816428846895485])
# ('softsign', [4.6764684628208215, 0.02236791361357501])
# ('relu', [4.569708056768027, 0.02911685306594678])
# ('tanh', [4.825452603017599, 0.01041264944080216])
# ('sigmoid', [4.646497667226559, 0.019861164674122637])
# ('hard_sigmoid', [4.77921117634742, 0.012148091014269186])
# ('linear', [15.96269532662306, 0.009641342074816815])
# """
# 
# ## Hyper Search for Dropout ##
# #%%
# dropout_results = []
# dropouts = [0.1, 0.2, 0.3]
# 
# for drop in dropouts:
#     new_mod, hist = generate_ffnn_model(dropout=drop)
#     scores = new_mod.evaluate_generator(create_validation_generator(image_size, 128))
#     dropout_results.append((drop, scores))
# 
# #%%
# for res in dropout_results:
#     print(res)
# """
# (0.1, [4.450355302124376, 0.03702275358453663])
# (0.2, [4.494139762169323, 0.0335518704261092])
# (0.3, [4.600360080352678, 0.021403779411840015])
# """
# 
# 
# ## Hyper Search for Learning Rate and Momentum ##
# #%%
# learn_rate = [0.0001, 0.001, 0.01, 0.1, 0.5]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# lr_mo_results = []
# 
# for lr in learn_rate:
#     for mo in momentum:
#         new_mod, hist = generate_ffnn_model(learn_rate=lr, momentum=mo)
#         scores = new_mod.evaluate_generator(create_validation_generator(image_size, 1))
#         lr_mo_results.append((lr, scores))
# 
# #%%
# for res in lr_mo_results:
#     print(res)
# 