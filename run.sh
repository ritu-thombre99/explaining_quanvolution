#!/bin/bash
python run_quanv.py --encoding angle --ansatz basic --filter_size 2
python run_quanv.py --encoding amplitude --ansatz basic --filter_size 2

python run_quanv.py --encoding angle --ansatz strong --filter_size 2
python run_quanv.py --encoding amplitude --ansatz strong --filter_size 2

python optimal_param_search.py >> param_search_result.txt

python train_qnn.py

python evaluate.py