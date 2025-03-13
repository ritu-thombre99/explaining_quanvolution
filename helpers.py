import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def cart2sph(x, y, z):
   r = np.sqrt(x**2 + y**2 + z**2) # r = sqrt(x² + y² + z²)
   if r == 0:
      return 0,0
   # normalize
   x,y,z = x/r, y/r, z/r
   # compute
   xy = np.sqrt(x**2 + y**2) # sqrt(x² + y²)
   theta = np.arctan2(y, x) 
   phi = np.arctan2(xy, z) 
   return theta, phi

def classwise_metrics(y_actual, y_pred, explanilibity, class_label):
   y_actual_new, y_pred_new = [],[]
   new_explanilibity = []
   for exp in explanilibity:
      if exp[0] == class_label:
         new_explanilibity.append(exp[1])
   for i in range(len(y_actual)):
      if y_actual[i] == class_label:
         y_actual_new.append(y_actual[i])
         y_pred_new.append(y_pred[i])

   acc = accuracy_score(y_actual_new,y_pred_new)
   f1 = f1_score(y_actual_new,y_pred_new, average='weighted')
   prec = precision_score(y_actual_new,y_pred_new, average='weighted')
   recall = recall_score(y_actual_new,y_pred_new, average='weighted')
   class_explanilibity = sum(new_explanilibity)/len(new_explanilibity)
   return acc, f1, prec, recall, class_explanilibity