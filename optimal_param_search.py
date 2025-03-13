
import tensorflow as tf
from tensorflow import keras
from itertools import product
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from scikeras.wrappers import KerasClassifier
from train_qnn import MyModel
from random import shuffle

train_test_split = 0.75

def param_search(encoding, ansatz, filter_size):
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

    parameters = {'batch_size': [32],
              'epochs': [100],
              'optimizer': ['adam', 'nadam', 'sgd']}
    grid_search = GridSearchCV(estimator = KerasClassifier(build_fn = MyModel, x_train = train_x, max_class_allowed = max_class_allowed), 
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 2,
                            verbose = 10)
    grid_search = grid_search.fit(train_x, train_y, verbose = 0)
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters: " + str(best_parameters))
    print("Best Score:",str(best_score))

if __name__ == "__main__":
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    kernel_sizes = [2]
    for encoding_type, ansatz_type, kernel_size in product(enocdings, ansatz, kernel_sizes):
        print("Current config:")
        print("Encoding:",encoding_type)
        print("Ansatz:",ansatz_type)
        print("Filter Size:",kernel_size)
        param_search(encoding_type, ansatz_type, kernel_size)