import matplotlib.pyplot as plt
from keras.utils import plot_model
#%%
def plot_loss_accuracy(history, accName, lossName):
  print(history.history.keys())
  #  "Accuracy"
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(accName)
  #plt.show()
  # "Loss"
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(lossName)
  #plt.show()

def plot_model_structure(model, filename):
  plot_model(model, to_file=filename)