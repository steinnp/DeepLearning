#%%
from keras import applications
from preprocessing import create_generator, create_validation_generator
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model 
from plot import plot_loss_accuracy

image_size = 150
batch_size = 20
print('Done importing')
#%%
#model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (image_size, image_size, 3))
model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (image_size, image_size, 3))

for layer in model.layers[:10]:
    layer.trainable = False
print('Done pretraining')
#%%
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
predictions = Dense(120, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.95, decay=1e-5, nesterov=True), metrics=["accuracy"])
#model_final.compile(loss = "categorical_crossentropy", optimizer = Adam(), metrics=["accuracy"])
history = model_final.fit_generator(create_generator(image_size, batch_size),
                    epochs=1200,
                    verbose=2,
                    validation_data=create_validation_generator(image_size, batch_size)
                    )
#plot_loss_accuracy(history)
print('Done training')