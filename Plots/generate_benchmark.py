import numpy as np
import pandas as pd
import json
import dataframe_image as dfi
import matplotlib.pyplot as plt

def plot_explainability(configurations, classes, explainability_list):
    x = np.arange(len(classes))  # x locations for the groups
    width = 0.5  # Bar width

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 Grid of Plots
    fig.suptitle('Comparison of Explainibility '+ r'$\mathcal{E}_{QNN}$' + ' per Class Across Configurations', fontsize=14)

    for i, ax in enumerate(axes.flat):
        explainibilities = explainability_list[i]

        bars = ax.bar(x, explainibilities, width, color='skyblue')

        ax.set_xlabel('Classes')
        ax.set_ylabel(r'$\mathcal{E}_{QNN}$')
        ax.set_title(configurations[i])
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.2*max(explainibilities))  # Accuracy range 0 to 1
        ax.grid(True, linestyle='--')

        # Show values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
    plt.savefig("Explainibilities.png" , bbox_inches='tight')


def plot_accuracy_and_f1(configurations, classes, accuracies_list, f1_scores_list):
    x = np.arange(len(classes))  # x locations for the groups
    width = 0.35  # Width of the bars

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 Grid of Plots
    fig.suptitle("Accuracy and F1-score per Class for Different Quanvolutions", fontsize=14)

    for i, ax in enumerate(axes.flat):
        accuracies = accuracies_list[i]
        f1_scores = f1_scores_list[i]

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-score', color='salmon')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title(configurations[i])
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.2)  # Scores range from 0 to 1
        ax.grid(True, linestyle='--')

        # Show values on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Move legend outside the last subplot
    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1,1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap
    plt.savefig("Accuracy-F1.png" , bbox_inches='tight')


def main():
    f = open('evaluate.json','r')
    results = json.load(f)
    f.close()

    df1 = []
    configurations = []
    accuracies_list = []
    f1_scores_list = []
    explainability_list = []
    classes = ['Jellyfish', 'Cat', 'Grasshopper', 'Dog']
    for config in results:
        dict1 = {}
        encoding, ansatz, _ = config.split(",")
        avg_metrics = results[config][1]
        dict1["Encoding"] = encoding
        dict1["Ansatz"] = ansatz
        dict1["Accuracy"] = np.round(avg_metrics[0],4)
        dict1["F1-Score"] = np.round(avg_metrics[1],4)
        dict1["Precision"] = np.round(avg_metrics[2],4)
        dict1["Recall"] = np.round(avg_metrics[3],4)
        dict1["Explainability"] = np.round(avg_metrics[4],4)
        df1.append(dict1)

        configurations.append("Encoding: "+encoding+" | "+"Ansatz: "+ansatz)
        class_metrics = results[config][2]

        accuracies_list.append([
                    np.round(class_metrics['0'][0],4), 
                    np.round(class_metrics['1'][0],4), 
                    np.round(class_metrics['2'][0],4), 
                    np.round(class_metrics['3'][0],4)
                ])
        f1_scores_list.append([
                    np.round(class_metrics['0'][1],4), 
                    np.round(class_metrics['1'][1],4), 
                    np.round(class_metrics['2'][1],4), 
                    np.round(class_metrics['3'][1],4)
                ])
        explainability_list.append([
                    np.round(class_metrics['0'][4],4), 
                    np.round(class_metrics['1'][4],4), 
                    np.round(class_metrics['2'][4],4), 
                    np.round(class_metrics['3'][4],4)
                ])
    

    df1 = pd.DataFrame(df1)
    df1.to_csv('main-result.csv',index = False)
    dfi.export(df1.style.hide(axis='index'), 'main-result.png')
    plot_accuracy_and_f1(configurations, classes, accuracies_list, f1_scores_list)
    plot_explainability(configurations, classes, explainability_list)

if __name__ == "__main__":
    main()