import argparse
import os
from glob import glob
from tqdm import tqdm
import pennylane as qml
import numpy as np
import numpy
from PIL import Image
import matplotlib.pyplot as plt
from itertools import product
from helpers import cart2sph

config = {}
config['encoding'] = 'angle'
config['ansatz'] = 'basic'
config['filter_size'] = 2



def make_quanv_filter(image, encoding, ansatz, filter_size):
    num_wires = filter_size**2
    dev = qml.device("default.qubit", wires=num_wires)
    @qml.qnode(dev)
    def circuit(patch):
        # Encoding
        patch = patch.reshape(patch.shape[0]**2, patch.shape[2])
        if encoding == 'angle':
            for patch_index in range(len(patch)):
                rot_angle = (patch[patch_index][0] + patch[patch_index][1] + patch[patch_index][2])/3
                qml.RY(np.pi * (rot_angle), wires = patch_index)
        elif encoding == 'amplitude':
            for patch_index in range(len(patch)):
                theta, phi = cart2sph(patch[patch_index][0], patch[patch_index][1], patch[patch_index][2])
                state = [np.sin(theta/2), np.exp(1j*phi)*np.cos(theta/2)]
                qml.StatePrep(state, wires=patch_index, normalize=True)

        # Ansatz
        if ansatz == 'basic':
            shape = qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=num_wires)
            qml.BasicEntanglerLayers(weights=np.zeros(shape), wires=range(num_wires))
        elif ansatz == 'strong':
            shape = qml.StronglyEntanglingLayers.shape(n_layers=num_wires//2, n_wires=num_wires)
            qml.StronglyEntanglingLayers(weights=np.zeros(shape), wires=range(num_wires))

        # Measurement producing (filter_size * filter_size) classical output values
        return [qml.expval(qml.PauliZ(patch_index)) for patch_index in range(len(patch))]
    
    # Quanvolute over the image
    out = []
    for i in range(0,image.shape[0], filter_size):
        row = []
        for j in range(0,image.shape[1], filter_size):
            row.append(circuit(image[i:i+filter_size, j:j+filter_size, :]))
        out.append(row)
    out = np.array(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding",type=str,
                        choices=['angle','amplitude'],help="choose the type of encoding to encode pixels into qubit. Default is angle")
    parser.add_argument("--ansatz",type=str,
                        choices=['basic','strong'],help="choose the type of ansatz to apply after encoding. Default is basic")
    parser.add_argument("--filter_size",choices=[2,3,5],type=int,help="Enter the quanvolution filter size (2x2,3x3,5x5). Default is 2x2")

    args = parser.parse_args()
    
    if args.encoding:
        config['encoding'] = args.encoding
    if args.ansatz:
        config['ansatz'] = args.ansatz
    if args.filter_size:
        config['filter_size'] = args.filter_size
    
    if config['encoding'] not in ['angle', 'amplitude']:
        print("Invalid encoding scheme. Allowed schemes: angle and amplitude")
        return
    if config['ansatz'] not in ['basic', 'strong']:
        print("Invalid ansatz. Allowed: basic and strong")
        return
    if config['filter_size'] not in [2,3,5]:
        print("Invalid filter size. Choose from [2x2, 3x3, 5x5]")
        return
    
    print("Running with config:", config)
    # find wherever JPEG images are and quanvolute them
    original_dir = "tiny-imagenet-200"
    jpeg_files = [y for x in os.walk(original_dir) for y in glob(os.path.join(x[0], '*.JPEG'))]
    for jpeg in tqdm(jpeg_files):
        image_path = '/'.join(jpeg.split("/")[:-1])
        image_name = jpeg.split("/")[-1].replace('.JPEG','')
        image = numpy.asarray(Image.open(jpeg).convert('RGB'))
        image = image / 255 # normalize
        quanv_output = make_quanv_filter(image, config['encoding'], config['ansatz'], config['filter_size'])
        np.save(image_path + "/" + image_name + "-" + config['encoding'] + "-" + config['ansatz'] + "-" + str(config['filter_size']) + ".npy", quanv_output)

if __name__ == "__main__":
    main()
    