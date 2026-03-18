# Setup env:
```
conda create --name explainable_qnn -c anaconda python=3.11.7 
conda activate explainable_qnn
conda install jupyter_server
conda install -n explainable_qnn nb_conda_kernels
conda install -n explainable_qnn ipykernel
```
# Steps to run code:

1. Generation quanvolutional features maps under different QNN setting. Stored as ```.npy``` in tiny-imagenet-200 directory:
```
python run_quanv.py --encoding angle --ansatz basic --filter_size 2
python run_quanv.py --encoding amplitude --ansatz basic --filter_size 2

python run_quanv.py --encoding angle --ansatz strong --filter_size 2
python run_quanv.py --encoding amplitude --ansatz strong --filter_size 2
```

2. Train 10 QNN models under 4-different configs (models saved as ```.h5``` in Models directory, training history saved as ```training_history.json``` in Plots directory), and evaluate their performance and dump their results in ```results.json``` in Plots directory:

```python train_qnns.py```

3. To get the plots (saved as png) and tables (saved as excel), run the following in Plots directory:

```python generate_plots.py```

### Auxiliary file: ```XAI.py``` is taken from [here](https://github.com/tavanaei/ExplainableCNN/blob/master/Code/XAI.py) required for generating ideal heatmaps from the [encoder-decoder based CNN](https://arxiv.org/pdf/2007.06712)
# QNN Setup:

- Quanvolution under different configuration:
    1. Entanglement type: BasicEntangling, StronglyEntangling
    2. Embedding type: Angle (rotation angles are averaged over RGB pixels), Amplitude (use RGB: (x,y,z) to encode with state-prep)

# Ideal heatmap Generation

- Use pre-trained explainable models from: **Explainable-CNN: https://github.com/tavanaei/ExplainableCNN/tree/master**

    "This paper proposes a new explainable convolutional neural network (XCNN) which represents important and driving visual features of stimuli in an end-to-end model architecture. This network employs encoder-decoder neural networks in a CNN architecture to represent regions of interest in an image based on its category"
    
    Paper: https://arxiv.org/pdf/2007.06712

    The heatmap in this paper are generated using iNNvestigate: https://arxiv.org/pdf/1808.04260 (https://github.com/albermax/innvestigate?tab=readme-ov-file)


# Note on reprodung plots:

- Depending on the platform, different (and incorrect) classwise accuracy, F1-score and explainibility plots might be generated since the class_label is indexed in one way while retrieving timy-imagenet-200 directory during training on one platform (where training took place) but they will be different if Plots are generated on a different system

- To accurately reproduce plots, run `train_qnns.py` before running `Plots/generate_plots.py`


References 
1. Grad-CAM tutorial: https://xai-tutorials.readthedocs.io/en/latest/_model_specific_xai/Grad-CAM.html
2. Grad-CAM implementation in TensorFlow keras: https://keras.io/examples/vision/grad_cam/
3. Quanvolution Neural net: https://pennylane.ai/qml/demos/tutorial_quanvolution
4. Survey on explainable AI: https://dl.acm.org/doi/10.1145/3563691
