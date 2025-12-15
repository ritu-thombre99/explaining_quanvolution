import tensorflow as tf
from random import shuffle
import os
import json
import numpy as np
from itertools import product
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from keras.models import load_model
from tqdm import tqdm
from skimage.transform import resize
from helpers import classwise_metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')
xcnn = torch.load('./Models/Model_TinyImageNet_128.net', map_location=torch.device('cpu')).to(device)
train_test_split = 0.7

def get_data(encoding, ansatz, filter_size):
    x_original, x_quanv ,y = [],[], []
    dirpath = './tiny-imagenet-200/train'
    wnids = os.listdir(dirpath) 
    for class_index, class_path in enumerate(wnids):
        images = os.listdir(dirpath + "/" + class_path + "/images/")
        for img in images:
            if img.endswith(".JPEG"):
                img_arr = np.asarray(Image.open(dirpath + "/" + class_path + "/images/"+img).convert('RGB'))
                x_original.append(img_arr)
                quanv_image_name = img.replace('.JPEG','')
                quanv_image_name = quanv_image_name + "-"+encoding+"-"+ansatz+"-"+str(filter_size)+".npy"
                if os.path.isfile(dirpath + "/" + class_path + "/images/"+quanv_image_name):
                    img_arr = np.load(dirpath + "/" + class_path + "/images/"+quanv_image_name)
                    x_quanv.append(img_arr)
                    y.append(class_index)
                else:
                    del x_original[-1]
    return x_original, x_quanv, np.array(y)

def get_xcnn_heatmap(image):
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    data = []
    data.append(transform_test(image[:,:,:]))
    ims = torch.stack(data).to(device)
    output = xcnn(ims)
    heatmap = xcnn.maps.cpu().detach().numpy()
    heatmap = heatmap[0].T.transpose(1,0,2)
    return heatmap

def grad_cam(q_model, x, class_channel):
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = q_model(x)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, x)
    # mean for each channel representing importance of that channel
    # https://stackoverflow.com/questions/58369040/when-and-why-do-we-use-tf-reduce-mean
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) 

    # multiply channel weights with channel matrices
    heatmap = x @ pooled_grads[..., tf.newaxis] 
    heatmap = tf.squeeze(heatmap) # flatten (1,31,31,1) -> (31,31)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

def calculate_explainability(heatmap_qnn, heatmap_xcnn):
    heatmap_qnn = resize(heatmap_qnn, heatmap_xcnn.shape, anti_aliasing=True)
    diff_map = abs(heatmap_qnn - heatmap_xcnn)
    explanilibity = np.linalg.norm(diff_map)
    if np.isnan(explanilibity):
        return -1
    return explanilibity

def caliberate_metrics(qnn, x_original, x_quanv, y, type = None):
    explanilibity = []
    sum_explanilibity = 0
    for i in tqdm(range(len(x_original))):
        heatmap_qnn = grad_cam(qnn, tf.convert_to_tensor([x_quanv[i]]), y[i])
        heatmap_xcnn = get_xcnn_heatmap(x_original[i])
        ret_val = int(calculate_explainability(heatmap_qnn, heatmap_xcnn))
        if ret_val != -1:
            explanilibity.append((int(y[i]), ret_val))
            sum_explanilibity = sum_explanilibity + ret_val

    predictions = [tf.argmax(pred).numpy() for pred in qnn.predict(tf.convert_to_tensor(x_quanv))]
    average_metrics = [
                            accuracy_score(y,predictions), 
                            f1_score(y,predictions, average='weighted'), 
                            precision_score(y,predictions, average='weighted'), 
                            recall_score(y,predictions, average='weighted'),
                            sum_explanilibity/len(explanilibity)
                        ]
    print("Type:", type)
    print("Acc:",average_metrics[0])
    print("F1:",average_metrics[1])
    print("Precision:",average_metrics[2])
    print("Recall:",average_metrics[3])
    print("Explainibility:",average_metrics[4])

    model_results = {}
    model_results["Type"] = type
    model_results["Encoding"] = encoding_type
    model_results["Ansatz"] = ansatz_type
    model_results["Iteration"] = curr_qnn
    model_results["Average Accuracy"] = average_metrics[0]
    model_results["Average F1-Score"] = average_metrics[1]
    model_results["Average Precision"] = average_metrics[2]
    model_results["Average Recall"] = average_metrics[3]
    model_results["Average Explainibility"] = average_metrics[4]

    for class_label in set(y):
        class_wise_metrics = classwise_metrics(y, predictions, explanilibity, class_label)
        model_results["Accuracy "+str(class_label)] = class_wise_metrics[0]
        model_results["F1-Score "+str(class_label)] = class_wise_metrics[1]
        model_results["Precision "+str(class_label)] = class_wise_metrics[2]
        model_results["Recall "+str(class_label)] = class_wise_metrics[3]
        model_results["Explainibility "+str(class_label)] = class_wise_metrics[4]

    with open("./Plots/results.json", "a") as f:
        json.dump(model_results, f)
        f.write("\n")
def compare_metrics(encoding_type, ansatz_type, curr_qnn, kernel_size = 2):
    qnn = load_model("./Models/qnn-"+ encoding_type + "_" + ansatz_type + "_" + str(curr_qnn) +".h5")
    x_original, x_quanv, y = get_data(encoding_type, ansatz_type, kernel_size)

    last_index_for_train = int(train_test_split*len(x_original))
    x_original_train, x_quanv_train, y_train = x_original[:last_index_for_train], x_quanv[:last_index_for_train], y[:last_index_for_train]
    x_original_test, x_quanv_test, y_test = x_original[last_index_for_train:], x_quanv[last_index_for_train:], y[last_index_for_train:]
    caliberate_metrics(qnn, x_original, x_quanv, y, type = "All")
    caliberate_metrics(qnn, x_original_train, x_quanv_train, y_train, type = "Train")
    caliberate_metrics(qnn, x_original_test, x_quanv_test, y_test, type = "Test")

    
if __name__ == "__main__":
    # Clean file to generate new results
    f = open("./Plots/results.json", "w")
    f.close()
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    num_qnns_trained = 10
    for encoding_type, ansatz_type, curr_qnn in product(enocdings, ansatz, range(num_qnns_trained)):
        print("Encoding:",encoding_type)
        print("Ansatz:",ansatz_type)
        print("QNN Iteration:",curr_qnn)
        compare_metrics(encoding_type, ansatz_type, curr_qnn)