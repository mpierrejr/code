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
            # if file_path.startswith(base_dir) is false then AssertionError

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
            # print(label, classes.index(label))

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

    # Y = np_utils.to_categorical(Y, len(classes))

    return X, Y, classes, processed_image_count


# ============================================

X, Y, classes, n = load_dataset('/home/mpierre/PycharmProjects/ImgArmadilhas/BaseOutubro/2_treinamento',224, False)

#teste_data_dir = ('/home/mpierre/PycharmProjects/ImgArmadilhas/dataset5/test_parametros/output')

# ============================================


from datetime import datetime
import errno

# name = os.path.join(".","runs_gesclerose", "test", "test_")
name = os.path.join("/", "home", "mpierre", "PycharmProjects","ImgArmadilhas" ,"BaseOutubro", "Treinamento")
date_time = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
name += date_time

path_img = os.path.join(name, "imgs")
path_models = os.path.join(name, "models")
path_tb = os.path.join(name, "logs_tensorboard")

print(name)
print(path_img)
print(path_models)

# Create folder if does not exist
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



    ##print(base_model.summary())

    # Adding custom Layers
    add_model = Sequential()
    add_model.add(Dense(1024, activation='relu', input_shape=base_model.output_shape[1:]))
    add_model.add(Dropout(0.6))
    add_model.add(Dense(2, activation='softmax'))
    # print(add_model.summary())

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    '''

    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    #Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    #top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False



    img_rows, img_cols, img_channel = 224, 224, 3
    base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling='avg',
                                                       input_shape=(img_rows, img_cols, img_channel))
    ##print(base_model.summary())

    # Adding custom Layers
    add_model = Sequential()
    add_model.add(Dense(1024, activation='relu', input_shape=base_model.output_shape[1:]))
    add_model.add(Dropout(0.6))
    add_model.add(Dense(2, activation='softmax'))
    #print(add_model.summary())




    #REDE VGG16

    #model.add(Conv2D(filters=64, input_shape=(224, 224, 3), kernel_size=(3, 3), \
    #                 activation='relu', padding='same'))

    #model.add(Conv2D(64, 3, 3, activation='relu', input_shape=(3, 224, 224), padding='same'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(64, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=64, input_shape=(224, 224, 3), kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(128, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(128, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),  padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(256, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(256, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(256, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(512, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(512, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(512, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(512, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(Conv2D(512, 3, 3, activation='relu', padding='same'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Conv2D(512, 3, 3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))



    #REDE ALEXNET
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), \
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())

    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())
    '''

    # Output Layer
    #model.add(Dense(2))
    #model.add(Activation('softmax'))

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

    # Confution Matrix and Classification Report
    validation_real_generator.reset()

    Y_pred = model.predict_generator(validation_real_generator, (samples // batch_size) + 1)

    y_pred = np.argmax(Y_pred, axis=1)

    # for i,j,k in zip(validation_real_generator.filenames,validation_real_generator.classes, y_pred):
    #    print ((i,j),k)

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

    # Clear model, and create it
    # model = None
    model = create_model("rmsprop")
    # evaluate best model

    # model = load_model("/home/mpierre/PycharmProjects/ImgArmadilhas/fc_model.h5")
    # model.load_weights("/home/mpierre/PycharmProjects/ImgArmadilhas/fc_model.h5")
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

    # Saving model...
    model.save(os.path.join(path_models, "last_modelFold" + str(index)))

    # evaluate the model
    scores = model.evaluate(xval, yval, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    # evaluate best model
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


#print (cvscores)

print("Melhores resultados")
#print (best_cvscores)

#IMPRESSAO DA MATRIX DE CONFUSAO

X_test, Y_test, classes_test, n_test = load_dataset('/home/mpierre/PycharmProjects/ImgArmadilhas/BaseOutubro/2_teste', 224, False)

ytrain = np_utils.to_categorical(Y_test, len(classes))

score = model.evaluate(X_test, ytrain, verbose=0)

print('\n', 'Test accuracy:', score[1]*100)

y_hat = model.predict(X_test)

ypreds = np.argmax(y_hat, axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, ypreds)

print(cm)

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,recall_score, precision_score, f1_score


#accuracyEval = accuracy_score(Y_test, ypreds)
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


print(history.history.keys())


