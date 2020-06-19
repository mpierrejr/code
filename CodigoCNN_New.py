import numpy as np
np.random.seed(2)

import os
from PIL import Image
import scipy.misc

from keras.utils import np_utils
import time as t
from datetime import datetime

print '#######################################################################################################################'
print 'Script para Classificacao CNN. Data de execucao:', datetime.now()
print '#######################################################################################################################'

# Tempo de execucao do codigo
tempoInicial = t.time()



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

    print('Quantidade de imagens para treinamento')
    print
    print ("Processed: %d images" % (processed_image_count))

    X = np.array(X, dtype='float32')
    Y = np.array(Y)

    print("Shuffle: " + str(shuffle))
    if shuffle:
        perm = np.random.permutation(len(Y))
        X = X[perm]
        Y = Y[perm]

    return X, Y, classes, processed_image_count


# DIRETORIO COM AS IMAGENS
X, Y, classes, n = load_dataset('/home/mpierre/PycharmProjects/ImgArmadilhas/BaseOutubro/2_treinamento',224, False)


# ============================================

from datetime import datetime
import errno

# DIRETORIO ONDE SERÃO SALVOS OS DADOS DE TREINAMENTO
name = os.path.join("/", "home", "mpierre", "PycharmProjects","ImgArmadilhas" ,"BaseOutubro", "Treinamento")
date_time = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
name += date_time

path_img = os.path.join(name, "imgs")
path_models = os.path.join(name, "models")
path_tb = os.path.join(name, "logs_tensorboard")

print(name)
print(path_img)
print(path_models)

try:
    os.makedirs(path_img)
    os.makedirs(path_models)
    os.makedirs(path_tb)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


# ============================================

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras import regularizers


def def_callbacks(index):
    callbacks = [

        ModelCheckpoint(os.path.join(path_models, "best_modelFold" + str(index)),
                        monitor='val_acc',
                        save_best_only=True,
                        mode='max',
                        verbose=0),
        CSVLogger(os.path.join(name, "log_Fold" + str(index) + ".csv"),
                  append=True,
                  separator=';'),
        TensorBoard(log_dir=os.path.join(path_tb, "Fold" + str(index)),
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
    ]

    return callbacks


# ============================================

from keras import applications
from keras.models import Model, load_model
from keras.layers import Activation

def create_model(optimizer_):
    dropout = 0.15
    bn_momentum = 0.4
    l2 = 0.0001
    l_r = 0.001
    # Model architecture definition
    weights_path = '../keras/examples/vgg16_weights.h5'
    top_model_weights_path = 'fc_model.h5'

    #model = Sequential()

    img_rows, img_cols, img_channel = 224, 224, 3

    # PRE TREINO COM DADOS DO IMAGENET
    #base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling='avg',
    #                                                   input_shape=(img_rows, img_cols, img_channel))

    base_model = applications.inception_v3.InceptionV3(include_top=False, weights=None, pooling='max',
                                                       input_shape=(img_rows, img_cols, img_channel))

    # REDE INCEPTION

    add_model = Sequential()
    add_model.add(Dense(1024, activation='relu', input_shape=base_model.output_shape[1:]))
    add_model.add(Dropout(0.6))
    add_model.add(Dense(2, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))



    if optimizer_ == "sgd":
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=l_r, decay=1e-6),
                      metrics=['accuracy'])
    if optimizer_ == "adam":
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=l_r, decay=1e-6),
                      metrics=['accuracy'])
    if optimizer_ == "rmsprop":
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=l_r, decay=1e-6),
                      metrics=['accuracy'])
    return model


# ============================================

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score

#img_width, img_height = 224, 224
img_width, img_height = 224, 224


def evaluate(model, eval_dir, samples):
    print ("Evaluation - " + eval_dir + " - " + "eval_name")

    validationReal_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_real_generator = validationReal_datagen.flow_from_directory(
        eval_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )


#============================================

    # Confution Matrix and Classification Report
    validation_real_generator.reset()

    Y_pred = model.predict_generator(validation_real_generator, (samples // batch_size) + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    print('\n')
    print ('##############################################')
    print('Matrix de Confusao')
    print ('##############################################')
    print

    confusionM = confusion_matrix(validation_real_generator.classes, y_pred)

    print(confusionM)
    print
    accuracyEval = (accuracy_score(validation_real_generator.classes, y_pred) * 100)
    recallEval = (recall_score(validation_real_generator.classes, y_pred) * 100)
    precisionEval = (precision_score(validation_real_generator.classes, y_pred) * 100)
    f1Eval = (f1_score(validation_real_generator.classes, y_pred) * 100)

    print ("Acuracia " + str(accuracyEval))
    print ("Recall " + str(recallEval))
    print ("Precisao " + str(precisionEval))
    print ("F1-Score " + str(f1Eval))
    print
    print('Classification Report')
    target_names = classes

    report = classification_report(validation_real_generator.classes, y_pred, target_names=target_names)

    print(report)

    outs_eval = open(name + '/evaluation_' + "eval_name" + '.txt', 'w+')

    outs_eval.write("Evaluation on %s samples\nAccuracy: %s\nConfusion Matrix\n%s\n\n%s" % (
        samples, accuracyEval, confusionM, report))


# ==================================================

from keras.models import load_model

def load_model_and_return_score(path, xval, yval):
    model = load_model(path)
    model.load_weights(path)
    scores = model.evaluate(xval, yval, verbose=0)
    return scores[1] * 100

def save_plots(history, index):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Acuracia')
    plt.ylabel('Acuracia')
    plt.xlabel('Epoca')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    fig.savefig(os.path.join(path_img, 'acc_Fold' + str(index) + '.png'), bbox_inches='tight', dpi=300)
    plt.close();


# ===================================================

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# ===================================================

batch_size = 32
nb_epoch = 25

# % matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

# plt.ioff()
cvscores = []
best_cvscores = []
K = 2
l_r = 0.0001

# Instantiate the cross validator
skf = StratifiedKFold(n_splits=K, shuffle=True)

for index, (train_indices, val_indices) in enumerate(skf.split(X, Y)):
    print('_________________________________________________________________________\n')
    print ("Training on fold " + str(index + 1) + "/" + str(K) + "...")
    # Generate batches from indices
    xtrain, xval = X[train_indices], X[val_indices]
    ytrain, yval = Y[train_indices], Y[val_indices]

    ytrain = np_utils.to_categorical(ytrain, len(classes))
    yval = np_utils.to_categorical(yval, len(classes))

    print('Training set: ' + str(xtrain.shape[0]) + ' images')
    print('Test set: ' + str(xval.shape[0]) + ' images')
    print('')

    steps = len(xtrain) * 2
    # steps = 2
    nb_test_samples = len(xval)

    model = create_model("rmsprop")

    callbacks = def_callbacks(index)

    optimizer_ = "rmsprop"

    if optimizer_ == "sgd":
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=l_r, decay=1e-6),  # antes sgd, adam, rmsprop
                      metrics=['accuracy'])
    if optimizer_ == "adam":
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=l_r, decay=1e-6),  # antes sgd, adam, rmsprop
                      metrics=['accuracy'])

    if optimizer_ == "rmsprop":
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=l_r, decay=1e-6),  # antes sgd, adam, rmsprop
                      metrics=['accuracy'])

    datagen.fit(xtrain)

    history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                                  epochs=nb_epoch,
                                  steps_per_epoch=50,
                                  validation_data=(xval, yval),
                                  callbacks=callbacks)

    #Salva o modelos

    model.save(os.path.join(path_models, "last_modelFold" + str(index)))
    scores = model.evaluate(xval, yval, verbose=0)

    cvscores.append(scores[1] * 100)

    best_score = load_model_and_return_score(os.path.join(path_models, "best_modelFold" + str(index)), xval, yval)
    best_cvscores.append(best_score)

    save_plots(history, index)


print
print ('##############################################')
print("Last models: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("Best models: %.2f%% (+/- %.2f%%)" % (np.mean(best_cvscores), np.std(best_cvscores)))

print ('##############################################')

tempoFim = t.time()
tempoDeExecucao = tempoFim - tempoInicial
tempoEmMinutos = tempoDeExecucao / 60
print('\n')
print ('##############################################')
print 'Tempo de execucao:', tempoEmMinutos, 'minuto(s).'
print ('##############################################')

print("Melhores resultados")

#TERMINO DO TREINAMENTO

