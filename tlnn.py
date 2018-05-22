#%%
from keras import applications
from preprocessing import create_generator, create_validation_generator
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model 
from plot import plot_loss_accuracy

image_size = 192
print('Done importing')
#%%
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (image_size, image_size, 3))

for layer in model.layers[:5]:
    layer.trainable = False
print('Done pretraining')
#%%
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(120, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
history = model_final.fit_generator(create_generator(image_size, 32),
                    epochs=120,
                    verbose=2,
                    validation_data=create_validation_generator(image_size, 16)
                    )
plot_loss_accuracy(history)
print('Done training')