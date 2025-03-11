# M1: Proposal

- Downloaded tine-imagenet from https://github.com/rmccorm4/Tiny-Imagenet-200?tab=readme-ov-file
- Found giant imagenet from https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description, did not download


### Setup env:
    # conda create --name explainable_qnn -c anaconda python=3.11.7 
    # conda activate explainable_qnn
    # conda install jupyter_server
    # conda install -n explainable_qnn nb_conda_kernels
    # conda install -n explainable_qnn ipykernel

References 
1. Grad-CAM tutorial: https://xai-tutorials.readthedocs.io/en/latest/_model_specific_xai/Grad-CAM.html
2. Grad-CAM implementation in TensorFlow keras: https://keras.io/examples/vision/grad_cam/
3. Quanvolution Neural net: https://pennylane.ai/qml/demos/tutorial_quanvolution
4. Survey on explainable AI: https://dl.acm.org/doi/10.1145/3563691


# M2: Midterm checkpoint:

- Use pre-trained explainable models from: **Explainable-CNN: https://github.com/tavanaei/ExplainableCNN/tree/master**

    "This paper proposes a new explainable convolutional neural network (XCNN) which represents important and driving visual features of stimuli in an end-to-end model architecture. This network employs encoder-decoder neural networks in a CNN architecture to represent regions of interest in an image based on its category"
    
    Paper: https://arxiv.org/pdf/2007.06712

    The heatmap in this paper are generated using iNNvestigate: https://arxiv.org/pdf/1808.04260 (https://github.com/albermax/innvestigate?tab=readme-ov-file)

- Quanvolution under different configuration:
    1. Filter size: 2x2, 3x3, 5x5
    2. Ansatz type: BasicEntangling, StronglyEntangling
    3. Embedding type: Angle (rotation angles are averaged over RGB pixels), Amplitude (use RGB: (x,y,z) to encode with state-prep)


# TODO for future checkpoints

Use ScoreCAM: non-graident based heatmap explainer 
    
    Paper: https://arxiv.org/pdf/1910.01279
    
    Code: https://github.com/tabayashi0117/Score-CAM/blob/master/Score-CAM.ipynb
