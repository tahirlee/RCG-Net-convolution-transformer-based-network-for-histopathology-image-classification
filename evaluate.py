import tensorflow as tf
import config
from prepare_data import get_datasets
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

train_generator, valid_generator, test_generator, \
train_num, valid_num, test_num= get_datasets()

model = tf.keras.models.load_model(config.trained_model_name)

np.set_printoptions(precision = 4)
num_batches = test_generator.samples//config.BATCH_SIZE

accm = np.zeros([num_batches])

for i in range(num_batches):

    print('testing batch number: {}/{}'.format(i + 1, num_batches))
    x, y = test_generator.next()
    yp = model.predict(x)
    ya = np.argmax(y, axis=1)
    ypa = np.argmax(yp, axis=1)
    if i == 0:
        ytrue = y;
        ypred = yp
    else:
        ytrue = np.concatenate((ytrue, y))
        ypred = np.concatenate((ypred, yp))
    n_true = np.where(ya == ypa)[0]
    accm[i] = len(n_true) / len(ya)
#################################################
preds = np.argmax(ypred,axis=1)
test_lab = np.zeros([ytrue.shape[0]])
for i in range(ytrue.shape[0]):
    test_lab[i] = np.where(ytrue[i,:]==1)[0]

from sklearn.metrics import precision_score, recall_score

accuracy_score = accuracy_score(test_lab, preds, normalize=True)

precision = precision_score(test_lab, preds , average='weighted')
recall = recall_score(test_lab, preds , average='weighted')

F1 = 2 * ((precision * recall) / (precision + recall))

print("--------------")
print(accuracy_score)
print(precision)
print(recall)
print(F1)
