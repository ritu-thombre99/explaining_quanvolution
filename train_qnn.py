
import tensorflow as tf
from tensorflow import keras
from itertools import product
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History 

train_test_split = 0.75
def MyModel(x_train, max_class_allowed, optimizer = 'nadam'):
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.Input(shape=(x_train[0].shape)),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(400, activation="sigmoid"),
        keras.layers.Dense(100, activation="sigmoid"),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(50, activation="sigmoid"),
        keras.layers.Dense(max_class_allowed, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_qnn_model(encoding, ansatz, filter_size, optimizer):
    train, test =  [],[]
    dirpath = './tiny-imagenet-200/train'
    wnids = os.listdir(dirpath) 
    max_class_allowed = len(wnids)
    for class_index, class_path in enumerate(wnids):
        data = []
        images = os.listdir(dirpath + "/" + class_path + "/images/")
        for img in images:
            if img.endswith("-"+encoding+"-"+ansatz+"-"+str(filter_size)+".npy"):
                # print(encoding+"-"+ansatz+"-"+str(filter_size))
                img = np.load(dirpath + "/" + class_path + "/images/"+img)
                data.append((img, class_index))
        
        last_index = int(train_test_split*len(data))
        train = train + data[:last_index]
        test = test + data[last_index:]

    shuffle(train)
    shuffle(test)

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
    q_model = MyModel(train_x, max_class_allowed, optimizer = optimizer)

    n_epochs = 500
    q_history = q_model.fit(
        train_x,
        train_y,
        validation_data=(test_x[:len(test_x)//4], test_y[:len(test_y)//4]),
        batch_size=32,
        epochs=n_epochs,
        verbose=2,
        callbacks=[history])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
    title = "Accuracy and Loss for " + encoding + " encoding " + ansatz + " ansatz"
    fig.suptitle(title)

    ax1.plot(q_history.history["accuracy"], label="Training Accuracy")
    ax1.plot(q_history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.grid()
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(q_history.history["loss"], label="Training Loss")
    ax2.plot(q_history.history["val_loss"], label="Validation Loss")
    ax2.set_ylabel("Loss")
    ax2.grid()
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig('./Plots/'+title+".png" , bbox_inches='tight')
    q_model.save("./Models/qnn-"+ encoding + "-" + ansatz + "-" + str(filter_size) +".h5")
        

if __name__ == "__main__":
    # optimal optimizers from GridSearchCV
    optimizer = {}
    optimizer[('angle','basic')] = 'nadam'
    optimizer[('angle','strong')] = 'adam'
    optimizer[('amplitude','basic')] = 'nadam'
    optimizer[('amplitude','strong')] = 'nadam'

    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    kernel_sizes = [2]
    for encoding_type, ansatz_type, kernel_size in product(enocdings, ansatz, kernel_sizes):
        train_qnn_model(encoding_type, ansatz_type, kernel_size, optimizer[(encoding_type, ansatz_type)])