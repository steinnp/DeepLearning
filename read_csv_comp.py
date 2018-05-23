#%%
import csv
import matplotlib.pyplot as plt

train_loss = []
train_acc = []

vali_loss = []
vali_acc = []

train_loss_f = []
train_acc_f = []

vali_loss_f = []
vali_acc_f = []

with open('tlnn_log.csv', encoding='utf-16') as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for row in reader:
        train_loss_f.append(float(row[0]))
        train_acc_f.append(float(row[1]))
        vali_loss_f.append(float(row[2]))
        vali_acc_f.append(float(row[3]))

with open('tlnn_fast_data.csv', encoding='utf-16') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_loss.append(float(row[0]))
        train_acc.append(float(row[1]))
        vali_loss.append(float(row[2]))
        vali_acc.append(float(row[3]))

#  "Accuracy"
plt.plot(train_acc_f)
plt.plot(vali_acc_f)
plt.plot(train_acc)
plt.plot(vali_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train (lr=0.0001)', 'validation (lr=0.0001)', 'train (lr=0.001)', 'validation (lr=0.001)'], loc='lower right')
plt.show()
# "Loss"
plt.plot(train_loss_f)
plt.plot(vali_loss_f)
plt.plot(train_loss)
plt.plot(vali_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train (lr=0.0001)', 'validation (lr=0.0001)', 'train (lr=0.001)', 'validation (lr=0.001)'], loc='upper right')
plt.show()