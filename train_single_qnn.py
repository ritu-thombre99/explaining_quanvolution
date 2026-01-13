
import tensorflow as tf
from tensorflow import keras
from itertools import product
from random import shuffle
import os, json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History 
from helpers import get_data, caliberate_metrics

train_test_split = 0.7
def MyModel(x_train, max_class_allowed):
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.Input(shape=(x_train[0].shape)),
        keras.layers.AveragePooling2D(pool_size=(2,2), strides=2),
        
        keras.layers.Flatten(),
        
        # First dense block
        keras.layers.Dense(400, kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.4),

        # Second dense block
        keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.4),

        # Third dense block
        keras.layers.Dense(50, kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(max_class_allowed, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_qnn_model(encoding, ansatz, filter_size, model_iter = None):
    train, test =  [],[]
    dirpath = './tiny-imagenet-200/train'
    wnids = os.listdir(dirpath) 
    max_class_allowed = len(wnids)
    data = []
    x_original, x_quanv, y = get_data(encoding, ansatz, filter_size)
    for img_index in range(len(x_original)):
        data.append((x_original[img_index], x_quanv[img_index] ,y[img_index]))
    
    shuffle(data)
    
    last_index = int(train_test_split*len(data))
    train = train + data[:last_index]
    test = test + data[last_index:]

    train_original, train_x, train_y = [], [], []
    test_original, test_x, test_y = [], [], []
    for train_item in train:
        train_original.append(train_item[0])
        train_x.append(train_item[1])
        train_y.append(train_item[2])

    for test_item in test:
        test_original.append(test_item[0])
        test_x.append(test_item[1])
        test_y.append(test_item[2])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    history = History()
    q_model = MyModel(train_x, max_class_allowed)

    n_epochs = 1000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=45, restore_best_weights=True)
    q_history = q_model.fit(
        train_x,
        train_y,
        validation_data=(test_x[:len(test_x)//2], test_y[:len(test_y)//2]),
        batch_size=32,
        epochs=n_epochs,
        verbose=2,
        callbacks=[history, early_stop])

    save_model_history = {}
    save_model_history["Encoding"] = encoding
    save_model_history["Ansatz"] = ansatz
    save_model_history["Iteration"] = model_iter
    save_model_history["Training Accuracy"] = q_history.history["accuracy"]
    save_model_history["Training Loss"] = q_history.history["loss"]
    save_model_history["Validation Accuracy"] = q_history.history["val_accuracy"]
    save_model_history["Validation Loss"] = q_history.history["val_loss"]

    f = open('Plots/training_history.json',"a")
    json.dump(save_model_history, f)
    f.close()

    caliberate_metrics(q_model, test_original, test_x, test_y, encoding, ansatz, model_iter,"Test")
    caliberate_metrics(q_model, train_original, train_x, train_y, encoding, ansatz, model_iter,"Train")
    q_model.save("./Models/qnn-"+ encoding + "_" + ansatz + "_" + str(model_iter) +".h5")

def train_curr_qnn(iter = None):
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    kernel_sizes = [2]
    for encoding_type, ansatz_type, kernel_size in product(enocdings, ansatz, kernel_sizes):
        train_qnn_model(encoding_type, ansatz_type, kernel_size, iter)