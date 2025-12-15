
import tensorflow as tf
from tensorflow import keras
from itertools import product
from random import shuffle
import os, json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History 

train_test_split = 0.7
def MyModel(x_train, max_class_allowed):
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.Input(shape=(x_train[0].shape)),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(400, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(50, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dense(max_class_allowed, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_qnn_model(encoding, ansatz, filter_size, model_iter = None):
    train, test =  [],[]
    dirpath = './tiny-imagenet-200/train'
    wnids = os.listdir(dirpath) 
    max_class_allowed = len(wnids)
    for class_index, class_path in enumerate(wnids):
        data = []
        images = os.listdir(dirpath + "/" + class_path + "/images/")
        for img in images:
            if img.endswith("-"+encoding+"-"+ansatz+"-"+str(filter_size)+".npy"):
                img = np.load(dirpath + "/" + class_path + "/images/"+img)
                data.append((img, class_index))
        
        last_index = int(train_test_split*len(data))
        train = train + data[:last_index]
        test = test + data[last_index:]

    train_x, train_y, test_x, test_y = [], [], [], []
    for train_item in train:
        train_x.append(train_item[0])
        train_y.append(train_item[1])

    for test_item in test:
        test_x.append(test_item[0])
        test_y.append(test_item[1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    history = History()
    q_model = MyModel(train_x, max_class_allowed)

    n_epochs = 750
    q_history = q_model.fit(
        train_x,
        train_y,
        validation_data=(test_x[:len(test_x)//2], test_y[:len(test_y)//2]),
        batch_size=32,
        epochs=n_epochs,
        verbose=2,
        callbacks=[history])

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
    q_model.save("./Models/qnn-"+ encoding + "_" + ansatz + "_" + str(model_iter) +".h5")

def train_curr_qnn(iter = None):
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    kernel_sizes = [2]
    for encoding_type, ansatz_type, kernel_size in product(enocdings, ansatz, kernel_sizes):
        train_qnn_model(encoding_type, ansatz_type, kernel_size, iter)