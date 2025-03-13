import tensorflow as tf
from random import shuffle
import os
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
    # shuffle(data)
    return x_original, x_quanv, np.array(y)

def get_xcnn_heatmap(image):
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    data = []
    data.append(transform_test(image[:,:,:]))
    ims = torch.stack(data).to(device)
    output = xcnn(ims)
    # prediction = int(torch.max(output.data, 1)[1].numpy()) # to use later to get XCNN accuracy
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

def compare_metrics(encoding_type, ansatz_type, kernel_size, f):
    qnn = load_model("./Models/qnn-"+ encoding_type + "-" + ansatz_type + "-" + str(kernel_size) +".h5")
    x_original, x_quanv, y = get_data(encoding_type, ansatz_type, kernel_size)

    explanilibity = []
    for i in tqdm(range(len(x_original))):
        heatmap_qnn = grad_cam(qnn, tf.convert_to_tensor([x_quanv[i]]), y[i])
        heatmap_xcnn = get_xcnn_heatmap(x_original[i])
        ret_val = int(calculate_explainability(heatmap_qnn, heatmap_xcnn))
        if ret_val != -1:
            explanilibity.append((int(y[i]), ))

    f.writelines(str(explanilibity)+"\n\n")

    predictions = [tf.argmax(pred).numpy() for pred in qnn.predict(tf.convert_to_tensor(x_quanv))]
    
    average_metrics = [
                            accuracy_score(y,predictions), 
                            f1_score(y,predictions, average='weighted'), 
                            precision_score(y,predictions, average='weighted'), 
                            recall_score(y,predictions, average='weighted')
                        ]

    print("Acc:",average_metrics[0])
    print("F1:",average_metrics[1])
    print("Precision:",average_metrics[2])
    print("Recall:",average_metrics[3])

    f.writelines(str(average_metrics)+"\n\n")
    class_wise_metrics = {}
    for class_label in set(y):
        class_wise_metrics[class_label] = classwise_metrics(y, predictions, class_label)
    f.writelines(str(class_wise_metrics)+"\n\n")
    
if __name__ == "__main__":
    f = open("./Plots/evaluate.txt","w")
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    kernel_sizes = [2]
    for encoding_type, ansatz_type, kernel_size in product(enocdings, ansatz, kernel_sizes):
        f.writelines(encoding_type + "," + ansatz_type + "," + str(kernel_size)+"\n\n")
        compare_metrics(encoding_type, ansatz_type, kernel_size, f)
    f.close()