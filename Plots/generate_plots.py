import pandas as pd
import json
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import dataframe_image as dfi
import os

wnid_to_class_label = {
                        "n01910747":"Jellyfish",
                        "n02106662":"Dog",
                        "n02124075":"Cat",
                        "n02226429":"Grasshopper"
                    }

# class label is name of the animal, index is class index
# index can vary depending on OS platform i.e. order in which directories are loaded
# to fix that we add the order in which directories are loaded
dirpath = '../tiny-imagenet-200/train'
wnids = os.listdir(dirpath) 
max_class_allowed = len(wnids)
class_label = []
for class_index, class_path in enumerate(wnids):
    class_label.append(wnid_to_class_label[class_path])


def plot_accuracy_f1_score(df, encoding_type, entanglement_type, result_type):
    temp_df = df[(df.Ansatz == entanglement_type) & (df.Encoding == encoding_type) & (df.Type == result_type)]

    accuracy_error, f1_score_error = [],[]
    accuracy, f1_score = [], []
    min_accuracy, min_f1_score = [], []
    max_accuracy, max_f1_score = [], []
    for class_index in range(len(class_label)):
        accuracy_error.append(np.std(list(temp_df['Accuracy '+str(class_index)])).item())
        f1_score_error.append(np.std(list(temp_df['F1-Score '+str(class_index)])).item())
    
        accuracy.append(temp_df["Accuracy "+str(class_index)].mean())
        f1_score.append(temp_df["F1-Score "+str(class_index)].mean())

        min_accuracy.append(temp_df["Accuracy "+str(class_index)].min())
        min_f1_score.append(temp_df["F1-Score "+str(class_index)].min())

        max_accuracy.append(temp_df["Accuracy "+str(class_index)].max())
        max_f1_score.append(temp_df["F1-Score "+str(class_index)].max())

    fig, ax = plt.subplots(figsize = (10,8))
    width = 0.35
    x = np.arange(len(class_label))

    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', yerr = accuracy_error, alpha=0.6)
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1-score', yerr = f1_score_error, alpha=0.6)

    ax.set_xlabel('Classes',fontsize=20)
    ax.set_ylabel('Scores',fontsize=20)
    ax.set_title(result_type+" Accuracy and F1-Score per class: Encoding: "+encoding_type+" | "+"Entanglement: "+entanglement_type,fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_label,fontsize=20)
    ax.set_ylim(0, 1.3)  # Scores range from 0 to 1
    ax.grid(True, linestyle='--')
    ax.legend(fontsize=20)

    # Show values on bars
    for i,bar in enumerate(bars1 + bars2):
        height = bar.get_height()
        y_pos = 0
        if i < 4:
            y_pos = max_accuracy[i]
        else:
            y_pos = max_f1_score[i%4]
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=20)
    plt.savefig(result_type + "-Accuracy-F1-" + str(encoding_type) + "-" + str(entanglement_type) + ".png" , bbox_inches='tight')

def plot_exp(df, encoding_type, entanglement_type, result_type):
    temp_df = df[(df.Ansatz == entanglement_type) & (df.Encoding == encoding_type) & (df.Type == result_type)]

    exp_error = []
    mean, min, max = [], [], []
    for class_index in range(len(class_label)):
        exp_error.append(np.std(list(temp_df['Explainibility '+str(class_index)])).item())
        mean.append(temp_df['Explainibility '+str(class_index)].mean())
        min.append(temp_df['Explainibility '+str(class_index)].min())
        max.append(temp_df['Explainibility '+str(class_index)].max())

    fig, ax = plt.subplots(figsize = (10,10))
    width = 0.5
    x = np.arange(len(class_label))

    bars1 = ax.bar(x - width/2, mean, width, yerr = exp_error, color = 'green',alpha=0.6)

    ax.set_xlabel('Classes',fontsize=20)
    ax.set_ylabel(r'$\mathcal{E}_{QNN}$',fontsize=20)
    ax.set_title(result_type + " Explainibility "+ r'$\mathcal{E}_{QNN}$' + " per class: Encoding: "+encoding_type+" | "+"Entanglement: "+entanglement_type,fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_label,fontsize=20)
    # ax.set_ylim(0, 40)  
    ax.grid(True, linestyle='--')

    # Show values on bars
    for i,bar in enumerate(bars1):
        height = bar.get_height()
        y_pos = max[i]
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=20)
    plt.savefig(result_type+"-Explainibility-" + str(encoding_type) + "-" + str(entanglement_type) + ".png" , bbox_inches='tight')

def average_metrics_table(df):
    result_df = pd.DataFrame(columns = ["Encoding", "Entanglement",
                                "Average Accuracy", "Stdev Accuracy",
                                "Average F1-Score", "Stdev F1-Score",
                                "Average Explainibility","Stdev Explainibility"])
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    for encoding_type, ansatz_type in product(enocdings, ansatz):
        temp_df = df[(df.Ansatz == ansatz_type) & (df.Encoding == encoding_type)]
        result_df.loc[len(result_df)] = [encoding_type, ansatz_type,
                    temp_df['Average Accuracy'].mean(), 
                    np.std(list(temp_df['Average Accuracy'])).item(),
                    temp_df['Average F1-Score'].mean(),
                    np.std(list(temp_df['Average F1-Score'])).item(),
                    temp_df['Average Explainibility'].mean(),
                    np.std(list(temp_df['Average Explainibility'])).item()
               ]
    result_df.to_excel("average_results.xlsx", index=False)
    dfi.export(result_df, "average_results.png")

def plot_metric(metric_array, ax, label, color):
    mean = np.mean(metric_array, axis=0)
    std = np.std(metric_array, axis=0)
    epochs = np.arange(len(mean))
    ax.plot(epochs, mean, label=label, color=color)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.3, color=color)

def plot_training_history(encoding_type, entanglement_type, training_acc, training_loss, val_acc, val_loss):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
    title = "Accuracy and Loss for Encoding: "+encoding_type+" | "+"Entanglement: "+entanglement_type
    fig.suptitle(title)

    # Plot Accuracy
    plot_metric(training_acc, ax1, 'Train Accuracy', 'blue')
    plot_metric(val_acc, ax1, 'Validation Accuracy', 'orange')
    ax1.set_ylabel("Accuracy")
    ax1.grid()
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Plot loss
    plot_metric(training_loss, ax2, 'Train Loss', 'blue')
    plot_metric(val_loss, ax2, 'Validation Loss', 'orange')
    ax2.set_ylabel("Loss")
    ax2.grid()
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.savefig("Accuracy-Loss-" + str(encoding_type) + "-" + str(entanglement_type) + ".png" , bbox_inches='tight')

if __name__ == "__main__":
    with open('results.json','r') as f:
        file_content = [json.loads(line) for line in f.readlines()]
    df = pd.DataFrame()
    for line in file_content:
        df = df._append(line, ignore_index=True)
    average_metrics_table(df)
    enocdings = ['angle','amplitude']
    ansatz = ['basic','strong']
    result_types = ['Train', 'Test']
    for encoding_type, ansatz_type, result_type in product(enocdings, ansatz, result_types):
        plot_accuracy_f1_score(df, encoding_type, ansatz_type, result_type)
        plot_exp(df, encoding_type, ansatz_type, result_type)

    f = open('training_history.json','r')
    lines = f.readlines()[0].split("}{")
    lines = lines[1:]
    lines = lines[:-1]
    f.close()
    file_content = [json.loads("{"+line+"}") for line in lines]
    for encoding_type, ansatz_type in product(enocdings, ansatz):
        training_loss, val_loss, training_acc, val_acc = [], [], [], []
        for line in file_content:
            if line["Encoding"] == encoding_type and line["Ansatz"] == ansatz_type:
                training_acc.append(line["Training Accuracy"])
                training_loss.append(line["Training Loss"])
                val_acc.append(line["Validation Accuracy"])
                val_loss.append(line["Validation Loss"])
        
        max_training_acc = max(len(row) for row in training_acc)
        training_acc = np.array([
            row + [np.nan] * (max_training_acc - len(row))
            for row in training_acc])

        max_training_loss = max(len(row) for row in training_loss)
        training_loss = np.array([
            row + [np.nan] * (max_training_loss - len(row))
            for row in training_loss])


        max_val_acc = max(len(row) for row in val_acc)
        val_acc = np.array([
            row + [np.nan] * (max_val_acc - len(row))
            for row in val_acc])

        max_val_loss = max(len(row) for row in val_loss)
        val_loss = np.array([
            row + [np.nan] * (max_val_loss - len(row))
            for row in val_loss])

        plot_training_history(encoding_type, ansatz_type, training_acc, training_loss, val_acc, val_loss)
    