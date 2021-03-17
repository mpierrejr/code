# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

import numpy as np
np.random.seed(2)

import os
from PIL import Image
import scipy.misc

from keras.utils import np_utils
import time as t
from datetime import datetime

# Tempo de execucao do codigo
tempoInicial = t.time()

# load model
model = load_model('/PATH')

#===============================================
# summarize model.
#model.summary()


def load_dataset(base_dir, img_size, shuffle=True):
    X = []
    Y = []
    processed_image_count = 0

    for root, subdirs, files in os.walk(base_dir):
        if subdirs:
            classes = subdirs

        for filename in files:
            file_path = os.path.join(root, filename)

            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip(os.sep)
            label = suffix.split(os.sep)[0]

            img = Image.open(file_path)
            img = img.convert('RGB')

            img = np.asarray(img)
            height, width, chan = img.shape

            assert chan == 3

            img = scipy.misc.imresize(img, size=(img_size, img_size), interp='bilinear')

            img = img / 255.

            X.append(img)
            Y.append(classes.index(label))

            processed_image_count += 1

    print('Quantidade de imagens para o teste final')
    print("Processadas: %d images" % (processed_image_count))
    print

    X = np.array(X, dtype='float32')
    Y = np.array(Y)

    print("Shuffle: " + str(shuffle))
    if shuffle:
        perm = np.random.permutation(len(Y))
        X = X[perm]
        Y = Y[perm]

    return X, Y, classes, processed_image_count


X_test, Y_test, classes_test, n_test = load_dataset('/PATH', 299, False)

ytrain = np_utils.to_categorical(Y_test, len(classes_test))

score = model.evaluate(X_test, ytrain, verbose=0)

print('\n', 'Test accuracy:', score[1]*100)


y_hat = model.predict(X_test)

ypreds = np.argmax(y_hat, axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, ypreds)

print(cm)

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,recall_score, precision_score, f1_score

accuracyEval = (accuracy_score(Y_test, ypreds) * 100)
recallEval = (recall_score(Y_test, ypreds) * 100)
precisionEval = (precision_score(Y_test, ypreds) * 100)
f1Eval = (f1_score(Y_test, ypreds) * 100)

print
print ("Acuracia " + str(accuracyEval))
print ("Recall " + str(recallEval))
print ("Precisao " + str(precisionEval))
print ("F1-Score " + str(f1Eval))
print

tempoFim = t.time()
tempoDeExecucao = tempoFim - tempoInicial
tempoEmMinutos = tempoDeExecucao / 60
print('\n')
print ('##############################################')
print 'Tempo de execucao:', tempoEmMinutos, 'minuto(s).'
print ('##############################################')


