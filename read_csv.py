#%%
import csv
import matplotlib.pyplot as plt

train_loss = []
train_acc = []

vali_loss = []
vali_acc = []

with open('tlnn_long_data.csv', encoding='utf-16') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_loss.append(float(row[0]))
        train_acc.append(float(row[1]))
        vali_loss.append(float(row[2]))
        vali_acc.append(float(row[3]))

#  "Accuracy"
plt.plot(train_acc)
plt.plot(vali_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(train_loss)
plt.plot(vali_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()