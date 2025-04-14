from train_single_qnn import train_curr_qnn
from tqdm import tqdm

# clear file to rewrite training data
f = open('Plots/training_history.json',"w")
f.close()

for qnn_i in tqdm(range(10)):
    train_curr_qnn(qnn_i)